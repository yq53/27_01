#include <asm-generic/errno.h>
#include <exception>
#include <iostream>

// n=0 -> 1; n=1 -> 1; n=2 -> 2; n=3 -> 3; 5; 8...
// 斐波那契

// 计算爬到第 n 阶楼梯的⽅法数（⽤迭代实现）
long long climbStairsIter(int n) {
    // 边界处理
    if (n >= 0 && n <= 40) {
        if (n == 0 || n == 1) {
            return 1;
        }

        // 前两次的方法数
        long long pre1 = 1;
        long long pre2 = 1;
        long long curr = 0;

        // 迭代一个个算
        for (int i = 2; i <= n; i++) {
            // 计算当前值
            curr = pre1 + pre2;

            // 更新
            pre1 = pre2;
            pre2 = curr;
        }

        return curr;
    }

    std::cout << "n超出边界" << std::endl;
    return -1;
}

// 计算爬到第 n 阶楼梯的⽅法数（⽤递归实现）
long long climbStairsRecur(int n) {
    // 边界处理
    if (n >= 0 && n <= 40) {
        if (n == 0 || n == 1) {
            return 1;
        }

        // 递归
        return climbStairsRecur(n - 1) + climbStairsRecur(n - 2);
    }

    std::cout << "n超出边界" << std::endl;
    return -1;
}

int main() {
    std::cout << "输⼊：⼀个整数 n（0 <= n <= 40）" << std::endl;

    // 初始化
    int n;
    std::cin >> n;

    std::cout << "====== 迭代函数调用 ======" << std::endl;
    long long iter = climbStairsIter(n);
    std::cout << "iter = " << iter << std::endl;

    std::cout << "====== 递归函数调用 ======" << std::endl;
    long long recur = climbStairsRecur(n);
    std::cout << "recur = " << recur << std::endl;

    // 结果比较
    if (iter == recur) {
        std::cout << "两函数调用结果一致" << std::endl;
    } else {
        std::cout << "两函数调用结果不一致" << std::endl;
    }
}