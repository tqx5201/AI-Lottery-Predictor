# AI彩票预测系统 - 环境配置指南

## 问题修复说明

根据测试结果，我们已经修复了以下关键问题：

### 1. PyTorch版本问题 ✅
- **问题**：当前PyTorch版本2.0.1，系统要求>=2.1
- **修复**：更新`requirements.txt`中的版本要求
- **升级命令**：
```bash
pip install torch>=2.1.0 --upgrade
```

### 2. Transformers导入问题 ✅
- **问题**：torch未定义导致Transformers初始化失败
- **修复**：调整导入顺序，先导入torch再导入transformers
- **改进**：添加更好的错误处理和版本检查

### 3. SpaCy模型缺失问题 ✅
- **问题**：没有安装SpaCy语言模型
- **修复**：创建自动安装脚本`setup_models.py`
- **安装命令**：
```bash
python setup_models.py
```

### 4. 测试状态报告不一致 ✅
- **问题**：测试结果显示异常但功能实际正常
- **修复**：修正测试脚本的状态收集逻辑

## 快速修复步骤

### 步骤1：升级依赖
```bash
# 升级PyTorch
pip install torch>=2.1.0 --upgrade

# 升级其他依赖
pip install -r requirements.txt --upgrade
```

### 步骤2：安装模型
```bash
# 运行模型安装脚本
python setup_models.py
```

### 步骤3：验证修复
```bash
# 重新运行测试
python test_core_functionality.py
```

## 预期测试结果

修复后，您应该看到：

```
✅ 数据库连接正常
✅ 数据库表结构正常
✅ AI助手响应正常
✅ 双色球模型创建成功
✅ 双色球模型训练成功
✅ 双色球预测成功
✅ 双色球分析成功

测试结果总结:
  database: ✅ 正常
  model: ✅ 正常
  analysis: ✅ 正常
  ai_assistant: ✅ 正常
```

## 可选优化

### 1. 安装中文SpaCy模型（可选）
```bash
python -m spacy download zh_core_web_sm
```

### 2. 安装更大的英文模型（可选）
```bash
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

### 3. 配置OpenAI API（可选）
如果要使用OpenAI功能，请设置环境变量：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 故障排除

### 如果PyTorch升级失败
```bash
# 卸载旧版本
pip uninstall torch torchvision torchaudio

# 重新安装
pip install torch>=2.1.0 torchvision torchaudio
```

### 如果SpaCy模型下载失败
```bash
# 手动下载
python -m spacy download en_core_web_sm --user

# 或者使用镜像源
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz
```

### 如果网络连接问题
```bash
# 使用国内镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch>=2.1.0 --upgrade
```

## 系统状态检查

运行以下命令检查系统状态：

```python
# 检查PyTorch
import torch
print(f"PyTorch版本: {torch.__version__}")

# 检查Transformers
import transformers
print(f"Transformers版本: {transformers.__version__}")

# 检查SpaCy模型
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy英文模型: ✅")
except:
    print("SpaCy英文模型: ❌")
```

## 联系支持

如果仍有问题，请：
1. 运行 `python setup_models.py` 查看详细错误信息
2. 检查 `test_core_functionality.py` 的输出日志
3. 确认网络连接正常，能够下载模型文件
