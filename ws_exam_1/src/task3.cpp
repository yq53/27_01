#include <iostream>

// 前序遍历：根 -> 左 -> 右
void preorderTraversal(const int* tree, int n, int i) {
    // 越界处理
    if (i >= n || tree[i] == -1) {
        return;
    }

    // root
    std::cout << tree[i] << " ";

    // left
    preorderTraversal(tree, n, 2 * i + 1); // 一直往下直到越界

    // right
    preorderTraversal(tree, n, 2 * i + 2); // 往下直到为空
}

// 中序遍历：左 -> 根 -> 右
void inorderTraversal(const int* tree, int n, int i) {
    // 越界处理
    if (i >= n || tree[i] == -1) {
        return;
    }

    // left
    inorderTraversal(tree, n, 2 * i + 1);

    // root
    std::cout << tree[i] << " ";

    // right
    inorderTraversal(tree, n, 2 * i + 2);
}

// 后序遍历：左 -> 右 -> 根
void postorderTraversal(const int* tree, int n, int i) {
    // 越界处理
    if (i >= n || tree[i] == -1) {
        return;
    }

    // left
    postorderTraversal(tree, n, 2 * i + 1);

    // right
    postorderTraversal(tree, n, 2 * i + 2);

    // root
    std::cout << tree[i] << " ";
}

// 构建⼀棵简单的测试树（填充数组）
void buildTestTree(int* tree, int n) {
    // 计算实际数组长度
    if (n < 7) {
        std::cout << "数组长度至少为7" << std::endl;
        return;
    }

    // 填充树
    tree[0] = 1;
    tree[1] = 2;
    tree[2] = 3;
    tree[3] = 4;
    tree[4] = 5;
    tree[5] = -1;
    tree[6] = -1;
}

int main() {
    const int n = 7;
    int tree[n];

    buildTestTree(tree, n);

    std::cout << "====== 前序遍历 ======" << std::endl;
    preorderTraversal(tree, n, 0);
    std::cout << std::endl;

    std::cout << "====== 中序遍历 ======" << std::endl;
    inorderTraversal(tree, n, 0);
    std::cout << std::endl;

    std::cout << "======后序遍历 ======" << std::endl;
    postorderTraversal(tree, n, 0);
    std::cout << std::endl;

    return 0;
}