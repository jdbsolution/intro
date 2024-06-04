


```markdown
## Python 基础教程

### 1. 安装 Python：

首先，你需要安装 Python 解释器。你可以从 [Python 官方网站](https://www.python.org/) 下载适合你操作系统的安装程序，并按照提示进行安装。

### 2. 编写你的第一个 Python 程序：

打开文本编辑器，输入以下内容：

```python
print("Hello, world!")
```

将文件保存为 `hello.py`，然后在命令行中执行以下命令：

```
python hello.py
```

你将看到输出 `Hello, world!`。

### 3. 变量和数据类型：

Python 中的变量不需要显式声明，直接赋值即可。示例：

```python
x = 5
name = "Alice"
```

常见的数据类型包括整数（int）、浮点数（float）、字符串（str）、列表（list）、元组（tuple）、字典（dictionary）等。

### 4. 控制流程：

#### 条件语句（if-elif-else）：

```python
x = 10
if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x is equal to 5")
else:
    print("x is less than 5")
```

#### 循环：

- `for` 循环：

```python
for i in range(5):
    print(i)
```

- `while` 循环：

```python
x = 0
while x < 5:
    print(x)
    x += 1
```

### 5. 函数：

```python
def greet(name):
    print("Hello, " + name + "!")

greet("Bob")
```

### 6. 列表和字典：

#### 列表：

```python
fruits = ["apple", "banana", "cherry"]
print(fruits[0])  # 输出 "apple"
fruits.append("orange")
print(fruits)  # 输出 ["apple", "banana", "cherry", "orange"]
```

#### 字典：

```python
person = {"name": "Alice", "age": 30}
print(person["name"])  # 输出 "Alice"
person["age"] = 31
print(person)  # 输出 {"name": "Alice", "age": 31}
```

### 7. 异常处理：

```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Error: Division by zero!")
```

以上是一个简单的 Python 基础教程，希望能够帮助你入门！如果你有任何问题，都可以问我。
```


