"""
forex_data_loader.py - 外汇数据下载和管理

支持：
- Dukascopy (免费，推荐)
- OANDA (需要API token)

使用方法:
    from forex_data_loader import DukascopyDataLoader

    loader = DukascopyDataLoader()
    data = loader.download_historical_data(
        pair='EURUSD',
        start_date='2023-01-01',
        end_date='2023-12-31',
        timeframe='1min'
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from pathlib import Path
import lzma
import struct
import time


class DukascopyDataLoader:
    """
    Dukascopy历史数据下载器

    数据格式：Tick级别 → 1分钟OHLCV聚合
    货币对：EUR/USD, GBP/USD, USD/JPY等
    """

    BASE_URL = "https://datafeed.dukascopy.com/datafeed"

    MAJOR_PAIRS = [
        'EURUSD',  # 欧元/美元（最流行）
        'GBPUSD',  # 英镑/美元
        'USDJPY',  # 美元/日元
        'USDCHF',  # 美元/瑞郎
        'AUDUSD',  # 澳元/美元
        'USDCAD',  # 美元/加元
        'NZDUSD',  # 纽元/美元
    ]

    def __init__(self, data_dir='data/forex'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 数据目录: {self.data_dir}")

    def download_tick_data(self, pair, year, month, day, hour):
        """
        下载指定时刻的tick数据

        URL格式:
        https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/01/00h_ticks.bi5
        """
        # Dukascopy月份从0开始
        month_str = f"{month:02d}"
        day_str = f"{day:02d}"
        hour_str = f"{hour:02d}"

        url = f"{self.BASE_URL}/{pair}/{year}/{month_str}/{day_str}/{hour_str}h_ticks.bi5"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 解压LZMA数据
            decompressed = lzma.decompress(response.content)

            # 解析二进制格式（每个tick 20字节）
            # 格式：timestamp(4), ask(4), bid(4), ask_volume(4), bid_volume(4)
            ticks = []
            for i in range(0, len(decompressed), 20):
                if i + 20 <= len(decompressed):
                    tick = struct.unpack('>IIIfI', decompressed[i:i+20])
                    ticks.append(tick)

            return ticks

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # 数据不存在（周末/节假日）
                return None
            else:
                print(f"HTTP错误 {url}: {e}")
                return None
        except Exception as e:
            print(f"下载失败 {url}: {e}")
            return None

    def ticks_to_ohlcv(self, ticks, timeframe='1min'):
        """
        将tick数据转换为OHLCV

        Args:
            ticks: List of (timestamp, ask, bid, ask_vol, bid_vol)
            timeframe: '1min', '5min', '15min', '1h'
        """
        if not ticks:
            return None

        df = pd.DataFrame(ticks, columns=['timestamp', 'ask', 'bid', 'ask_vol', 'bid_vol'])

        # 转换价格（Dukascopy格式：整数×1e-5）
        df['ask'] = df['ask'] * 1e-5
        df['bid'] = df['bid'] * 1e-5

        # 中间价
        df['price'] = (df['ask'] + df['bid']) / 2

        # Spread
        df['spread'] = df['ask'] - df['bid']

        # 时间戳转换（Dukascopy epoch: 1970-01-01）
        base_time = datetime(1970, 1, 1)
        df['datetime'] = df['timestamp'].apply(lambda x: base_time + timedelta(milliseconds=int(x)))
        df.set_index('datetime', inplace=True)

        # 聚合为OHLCV
        timeframe_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '1h': '1H',
        }

        freq = timeframe_map.get(timeframe, '1T')

        ohlcv = pd.DataFrame()
        ohlcv['open'] = df['price'].resample(freq).first()
        ohlcv['high'] = df['price'].resample(freq).max()
        ohlcv['low'] = df['price'].resample(freq).min()
        ohlcv['close'] = df['price'].resample(freq).last()
        ohlcv['volume'] = df[['ask_vol', 'bid_vol']].sum(axis=1).resample(freq).sum()
        ohlcv['spread'] = df['spread'].resample(freq).mean()

        return ohlcv.dropna()

    def download_historical_data(self, pair, start_date, end_date, timeframe='1min'):
        """
        下载历史数据范围

        Args:
            pair: 'EURUSD', 'GBPUSD'等
            start_date: '2020-01-01'
            end_date: '2024-01-01'
            timeframe: '1min', '5min', '15min'
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        all_data = []
        current = start

        print(f"\n下载 {pair} 数据: {start_date} → {end_date}")
        print(f"时间框架: {timeframe}")

        total_days = (end - start).days
        days_processed = 0

        while current <= end:
            year = current.year
            month = current.month - 1  # Dukascopy月份从0开始
            day = current.day

            day_data = []

            # 每小时下载一次
            for hour in range(24):
                ticks = self.download_tick_data(pair, year, month, day, hour)

                if ticks:
                    ohlcv = self.ticks_to_ohlcv(ticks, timeframe)
                    if ohlcv is not None and len(ohlcv) > 0:
                        day_data.append(ohlcv)

                # 避免请求过快
                time.sleep(0.05)

            # 合并当天数据
            if day_data:
                day_df = pd.concat(day_data).sort_index()
                all_data.append(day_df)

            current += timedelta(days=1)
            days_processed += 1

            # 进度报告
            if days_processed % 7 == 0 or current >= end:
                progress = days_processed / total_days * 100
                total_bars = sum(len(d) for d in all_data)
                print(f"  进度: {progress:.1f}% ({days_processed}/{total_days}天) - {total_bars} bars")

        # 合并所有数据
        if all_data:
            df = pd.concat(all_data).sort_index()

            # 去重（可能有重叠）
            df = df[~df.index.duplicated(keep='first')]

            # 保存到本地
            output_file = self.data_dir / f"{pair}_{timeframe}_{start_date}_{end_date}.csv"
            df.to_csv(output_file)
            print(f"\n✅ 保存: {output_file}")
            print(f"   数据量: {len(df)} bars")
            print(f"   时间范围: {df.index[0]} → {df.index[-1]}")
            print(f"   列: {list(df.columns)}")

            return df

        else:
            print(f"\n❌ 未能下载任何数据")
            return None

    def load_cached_data(self, pair, start_date, end_date, timeframe='1min'):
        """加载已缓存的数据"""
        cache_file = self.data_dir / f"{pair}_{timeframe}_{start_date}_{end_date}.csv"

        if cache_file.exists():
            print(f"✅ 从缓存加载: {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"   数据量: {len(df)} bars")
            return df
        else:
            print(f"⚠️  缓存不存在: {cache_file}")
            return None


# 快速开始示例
if __name__ == '__main__':
    print("="*80)
    print("Dukascopy外汇数据下载器")
    print("="*80)

    loader = DukascopyDataLoader()

    # 示例：下载EUR/USD 2024年1月数据（1分钟K线）
    pair = 'EURUSD'
    start = '2024-01-01'
    end = '2024-01-07'  # 先测试1周数据
    timeframe = '1min'

    print(f"\n开始下载 {pair} 数据...")
    print(f"  日期范围: {start} → {end}")
    print(f"  时间框架: {timeframe}")
    print(f"  预计时间: ~2-5分钟（取决于网速）")
    print()

    # 检查缓存
    data = loader.load_cached_data(pair, start, end, timeframe)

    if data is None:
        # 下载新数据
        data = loader.download_historical_data(pair, start, end, timeframe)

    if data is not None:
        print("\n" + "="*80)
        print("数据预览:")
        print("="*80)
        print(data.head(10))
        print(f"\n数据统计:")
        print(data.describe())
        print(f"\n缺失值:")
        print(data.isnull().sum())
    else:
        print("\n❌ 数据下载失败")
