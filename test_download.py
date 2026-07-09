#!/usr/bin/env python
"""测试 A股数据下载的各个步骤，诊断问题"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import akshare as ak
import pandas as pd
import time

print("测试 1: 检查 akshare 版本和简单调用")
print(f"akshare version: {ak.__version__}")
try:
    df = ak.stock_info_a_code_name()
    print(f"  OK: stock_info_a_code_name() 返回 {len(df)} 行")
    print(f"  列名: {df.columns.tolist()}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n测试 2: 测试 stock_a_industry_name_em (可能很慢)")
try:
    print("  正在调用 stock_a_industry_name_em() ...")
    start = time.time()
    df_ind = ak.stock_a_industry_name_em()
    elapsed = time.time() - start
    print(f"  OK: 返回 {len(df_ind)} 行, 耗时 {elapsed:.1f} 秒")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n测试 3: 测试获取 HS300 成分股")
try:
    print("  正在调用 index_stock_cons_csindex(symbol='000300') ...")
    start = time.time()
    df_hs300 = ak.index_stock_cons_csindex(symbol="000300")
    elapsed = time.time() - start
    print(f"  OK: 返回 {len(df_hs300)} 行, 耗时 {elapsed:.1f} 秒")
    print(f"  列名: {df_hs300.columns.tolist()}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n测试 4: 测试下载单只股票的历史数据")
try:
    print("  正在调用 stock_zh_a_hist(symbol='600519') ...")
    start = time.time()
    df_hist = ak.stock_zh_a_hist(
        symbol="600519",
        period="daily",
        start_date="20240101",
        end_date="20240131",
        adjust="",
    )
    elapsed = time.time() - start
    print(f"  OK: 返回 {len(df_hist)} 行, 耗时 {elapsed:.1f} 秒")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n所有测试完成")
