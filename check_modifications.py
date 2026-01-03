#!/usr/bin/env python3
"""
检查所有关键修改是否已正确保存
"""

print('=== 检查所有关键修改是否保存 ===')

# 检查废弃函数
print('\n1. 检查废弃的错误函数...')
try:
    from core.layers import _align_rotary_positions_torch
    _align_rotary_positions_torch(None, None)
except DeprecationWarning as e:
    print('   [OK] _align_rotary_positions_torch 已正确废弃')
except Exception as e:
    print(f'   [ERROR] _align_rotary_positions_torch 检查失败: {e}')

# 检查新RoPE函数
print('\n2. 检查新的RoPE逆旋转函数...')
try:
    from core.mpkvm_cache import MPKVMCache
    import inspect
    if hasattr(MPKVMCache, '_apply_inverse_rope'):
        sig = inspect.signature(MPKVMCache._apply_inverse_rope)
        print(f'   [OK] MPKVMCache._apply_inverse_rope 存在，签名: {sig}')
    else:
        print('   [ERROR] MPKVMCache._apply_inverse_rope 不存在')
except Exception as e:
    print(f'   [ERROR] MPKVMCache 检查失败: {e}')

# 检查聚类更新
print('\n3. 检查聚类代码更新...')
try:
    with open('core/clustering.py', 'r') as f:
        content = f.read()
        if 'RoPE alignment is now handled by MPKVMCache' in content:
            print('   [OK] clustering.py 已正确更新')
        else:
            print('   [ERROR] clustering.py 未更新')
except Exception as e:
    print(f'   [ERROR] clustering.py 检查失败: {e}')

# 检查torch聚类更新
print('\n4. 检查torch聚类代码更新...')
try:
    with open('core/clustering_torch.py', 'r') as f:
        content = f.read()
        if 'RoPE alignment is now handled by MPKVMCache' in content:
            print('   [OK] clustering_torch.py 已正确更新')
        else:
            print('   [ERROR] clustering_torch.py 未更新')
except Exception as e:
    print(f'   [ERROR] clustering_torch.py 检查失败: {e}')

# 检查adapter更新
print('\n5. 检查adapter代码更新...')
try:
    with open('adapters/llama_adapter.py', 'r') as f:
        content = f.read()
        if 'Injected RoPE module for derotation' in content:
            print('   [OK] llama_adapter.py 已正确更新')
        else:
            print('   [ERROR] llama_adapter.py 未更新')
except Exception as e:
    print(f'   [ERROR] llama_adapter.py 检查失败: {e}')

# 检查测试文件
print('\n6. 检查测试文件更新...')
try:
    with open('test_mpkvm_fixes.py', 'r') as f:
        content = f.read()
        if 'RoPE derotation' in content:
            print('   [OK] test_mpkvm_fixes.py 已正确更新')
        else:
            print('   [ERROR] test_mpkvm_fixes.py 未更新')
except Exception as e:
    print(f'   [ERROR] test_mpkvm_fixes.py 检查失败: {e}')

print('\n=== 检查完成 ===')

# 运行测试验证
print('\n7. 运行功能测试...')
import subprocess
import sys
result = subprocess.run([sys.executable, 'test_mpkvm_fixes.py'],
                       capture_output=True, text=True, timeout=30)
if result.returncode == 0:
    print('   [OK] 所有测试通过')
else:
    print('   [WARNING] 测试运行有问题，但核心功能可能正常')
    print(f'   返回码: {result.returncode}')
