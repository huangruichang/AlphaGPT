#!/usr/bin/env python
"""测试修复后的 download_stock_basic 函数"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.ashare_download import download_stock_basic, get_stock_pool

print("测试 download_stock_basic 函数")
print("=" * 60)

# 获取小的股票池进行测试
print("\n1. 获取股票池（前5只）...")
ts_codes = get_stock_pool(use_index=True)[:5]
print(f"   测试股票代码: {ts_codes}")

print("\n2. 下载股票基础信息...")
try:
    df_basic = download_stock_basic(ts_codes)
    print(f"   OK: 成功下载 {len(df_basic)} 行")
    print(f"   列名: {df_basic.columns.tolist()}")
    print(f"\n   前3行数据:")
    print(df_basic.head(3))
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")
