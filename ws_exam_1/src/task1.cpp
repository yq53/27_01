#include <iostream>

// 原地反转数组：将 arr 中的元素顺序颠倒
// arr 是数组指针，n 是数组⻓度
void reverseArray(int* arr, int n) {
    if (n >= 1 && n <= 100) {
        // 左右指针
        int* left = arr;
        int* right = arr + n - 1;

        // 遍历
        while (left < right) {
            int temp = *left;
            *left = *right;
            *right = temp;

            // 指针移动
            left++;
            right--;
        }

        // 结果输出
        std::cout << "输出：" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << *(arr + i) << " ";
        }

        return;
    }

    std::cout << "n取值范围为[1, 100]" << std::endl;
    return;
}

int main() {
    std::cout << "输⼊：\n"
              << " 1. 第⼀⾏⼀个整数 n（1 <= n <= 100）\n"
              << " 2. 第⼆⾏n个整数" << std::endl;

    // init
    int n;
    std::cin >> n;

    int arr[n - 1];
    for (int i = 0; i < n; i++) {
        std::cin >> arr[i];
    }

    reverseArray(arr, n);

    return 0;
}
