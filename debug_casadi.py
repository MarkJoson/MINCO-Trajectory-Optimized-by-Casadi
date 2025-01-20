import casadi as ca

# 1. 基本的打印辅助函数
def print_matrix(matrix, name="Matrix"):
    """更易读的矩阵打印函数"""
    print(f"\n{name}:")
    if isinstance(matrix, ca.MX):
        rows, cols = matrix.shape
        for i in range(rows):
            row_str = "["
            for j in range(cols):
                elem = matrix[i,j]
                # 尝试简化表达式
                if elem.is_constant():
                    row_str += f"{float(elem):8.3f}"
                else:
                    row_str += f"{str(elem):8s}"
                row_str += " "
            row_str += "]"
            print(row_str)

# 2. 带有详细信息的矩阵分析
def analyze_matrix(matrix, name="Matrix"):
    """详细分析矩阵结构"""
    print(f"\n=== Analysis of {name} ===")
    print(f"Shape: {matrix.shape}")
    print(f"Number of elements: {matrix.numel()}")
    print(f"Symbolic variables: {[str(v) for v in ca.symvar(matrix)]}")

    # 打印每个元素的详细信息
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            elem = matrix[i,j]
            print(f"\nElement [{i},{j}]:")
            print(f"  Expression: {str(elem)}")
            print(f"  Is symbolic: {elem.is_symbolic()}")
            print(f"  Is constant: {elem.is_constant()}")

# 4. 结构化展示复杂矩阵
def print_structured_matrix(matrix):
    """结构化展示矩阵内容"""
    rows, cols = matrix.shape

    # 获取元素的最大字符长度
    max_length = 0
    for i in range(rows):
        for j in range(cols):
            max_length = max(max_length, len(str(matrix[i,j])))

    # 打印矩阵
    print("\nMatrix structure:")
    print("─" * (cols * (max_length + 3) + 1))
    for i in range(rows):
        print("│", end=" ")
        for j in range(cols):
            elem = str(matrix[i,j])
            print(f"{elem:{max_length}}", end=" │ ")
        print("\n" + "─" * (cols * (max_length + 3) + 1))


# 1. 使用substitute方法替换符号变量
def mx_to_dense_substitute(mx_matrix, var_values=None):
    """
    通过substitute方法转换MX为Dense
    var_values: 字典，包含变量的值
    """
    if var_values is None:
        return mx_matrix

    # 获取所有符号变量
    vars = ca.symvar(mx_matrix)

    # 准备替换值
    result = mx_matrix
    for var in vars:
        if str(var) in var_values:
            result = ca.substitute(result, var, var_values[str(var)])

    return result