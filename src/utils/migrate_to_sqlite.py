#!/usr/bin/env python3
"""
数据迁移脚本：从JSON文件迁移到SQLite数据库
使用方法：python migrate_to_sqlite.py
"""

import os
import sys
import json
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.database_manager import DatabaseManager
except ImportError:
    from ..core.database_manager import DatabaseManager, DataMigration
from database_adapter import DatabaseAdapter


def main():
    """主迁移函数"""
    print("🚀 开始数据迁移：JSON → SQLite")
    print("=" * 50)
    
    # 检查JSON文件是否存在
    json_file = os.path.join(os.path.dirname(__file__), "history_data", "history_cache.json")
    
    if not os.path.exists(json_file):
        print("❌ 未找到现有的JSON缓存文件")
        print(f"   文件路径: {json_file}")
        print("   如果这是首次使用，可以忽略此消息")
        return
    
    print(f"📁 发现JSON文件: {json_file}")
    
    try:
        # 显示JSON文件信息
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        print(f"📊 JSON文件包含 {len(json_data)} 条缓存记录")
        
        # 初始化数据库管理器
        print("🔧 初始化SQLite数据库...")
        db_manager = DatabaseManager()
        
        # 执行数据迁移
        print("📦 开始迁移数据...")
        migration = DataMigration(db_manager, json_file)
        migration.migrate_from_json()
        
        # 显示迁移结果
        print("\n📈 迁移完成！数据库统计信息：")
        stats = db_manager.get_database_stats()
        
        if 'cache_count' in stats:
            print(f"   缓存记录数: {stats['cache_count']}")
        
        if 'history_data' in stats:
            for lottery_type, count in stats['history_data'].items():
                print(f"   {lottery_type}历史数据: {count}条")
        
        if 'db_size' in stats:
            size_mb = stats['db_size'] / (1024 * 1024)
            print(f"   数据库大小: {size_mb:.2f} MB")
        
        # 清理资源
        db_manager.close()
        
        print("\n✅ 数据迁移成功完成！")
        print("💡 原JSON文件已备份，可以安全删除")
        
        # 测试数据库适配器
        print("\n🧪 测试数据库适配器...")
        adapter = DatabaseAdapter()
        db_info = adapter.get_database_info()
        print(f"   适配器测试通过: {db_info.get('cache_count', 0)} 条缓存记录")
        adapter.close()
        
    except Exception as e:
        print(f"❌ 数据迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 迁移流程全部完成！")
    return True


def backup_json_file():
    """备份原JSON文件"""
    json_file = os.path.join(os.path.dirname(__file__), "history_data", "history_cache.json")
    
    if os.path.exists(json_file):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{json_file}.backup.{timestamp}"
        
        import shutil
        shutil.copy2(json_file, backup_file)
        print(f"📋 JSON文件已备份: {backup_file}")
        return backup_file
    
    return None


def show_database_info():
    """显示数据库信息"""
    try:
        print("\n📊 数据库详细信息：")
        print("-" * 30)
        
        adapter = DatabaseAdapter()
        db_info = adapter.get_database_info()
        
        print(f"缓存记录数: {db_info.get('cache_count', 0)}")
        print(f"预测记录数: {db_info.get('prediction_count', 0)}")
        
        if 'history_data' in db_info:
            print("历史数据统计:")
            for lottery_type, count in db_info['history_data'].items():
                print(f"  - {lottery_type}: {count}条")
        
        if 'db_size_formatted' in db_info:
            print(f"数据库大小: {db_info['db_size_formatted']}")
        
        adapter.close()
        
    except Exception as e:
        print(f"获取数据库信息失败: {e}")


if __name__ == "__main__":
    print("🎯 AI彩票预测系统 - 数据库迁移工具")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 执行迁移
    success = main()
    
    if success:
        # 显示详细信息
        show_database_info()
        
        print("\n💡 下一步:")
        print("1. 启动主程序测试数据库功能")
        print("2. 确认数据正常后可删除备份的JSON文件")
        print("3. 享受更快的数据存储和查询性能！")
    else:
        print("\n❌ 迁移失败，请检查错误信息并重试")
        
    print("\n按回车键退出...")
    input()
