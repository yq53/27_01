#include <iostream>

const int MAX_STUDENTS = 100;

// 1. 初始化成绩数组（n为学⽣⼈数，scores为学⽣分数）
void initScores(int* scores, int n) {
    for (int i = 0; i < n; i++) {
        scores[i] = -1;
    }
}

// 2. 录⼊成绩
void inputScores(int* scores, int n) {
    // 越界处理
    if (n > 0 && n <= MAX_STUDENTS) {
        for (int i = 0; i < n; i++) {
            std::cout << "请输入第 " << i + 1 << " 个学生的成绩：" << std::endl;
            std::cin >> scores[i];

            // 保证成绩范围，并且无需从头再次输入
            while (scores[i] < 0 || scores[i] > 100) {
                std::cout << "成绩范围应为[0, 100]，请重新输入：" << std::endl;
                std::cin >> scores[i];
            }
        }

        return;
    }

    std::cout << "n发生越界" << std::endl;
    return;
}

// 3. 计算平均分
double calcAverage(const int* scores, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += scores[i];
    }

    return sum / n;
}

// 4. 安全查询
// ⽬的：确保任何查询都不会导致程序崩溃
// 返回值：如果下标越界，打印“下标越界”并返回 -1；如果下标合法但成绩未录⼊，也返回 -1。
int safeGet(const int* scores, int n, int index) {
    // 下标越界
    if (index < 0 || index >= n) {
        std::cout << "下标越界" << std::endl;
        return -1;
    }

    // 成绩未录入
    if (scores[index] == -1) {
        return -1;
    }

    return scores[index];
}

// 5. 统计分数段（counts为各个分数段的个数）
// counts[0]：0~19
// counts[1]：20~39
// counts[2]：40~59
// counts[3]：60~79
// counts[4]：80~100
void countGrades(const int* scores, int n, int* counts) {
    for (int i = 0; i < n; i++) {
        if (scores[i] < 19) {
            counts[0]++;
        } else if (scores[i] < 39) {
            counts[1]++;
        } else if (scores[i] < 59) {
            counts[2]++;
        } else if (scores[i] < 79) {
            counts[3]++;
        } else {
            counts[4]++;
        }
    }
}

// 6. 找最⾼分与最低分
void findMinMax(const int* scores, int n, int* max, int* min) {
    *max = *min = scores[0];

    for (int i = 0; i < n; i++) {
        *max = (scores[i] > *max) ? scores[i] : *max;
        *min = (scores[i] < *min) ? scores[i] : *min;
    }
}

// 7.打印所有结果
void printResults(const int* scores, int n, double avg, const int* counts, int max, int min) {
    std::cout << "所有学生成绩：";

    for (int i = 0; i < n; i++) {
        std::cout << scores[i] << " ";
    }

    std::cout << std::endl;

    std::cout << "平均分：" << avg << std::endl;
    std::cout << "最高分：" << max << std::endl;
    std::cout << "最低分：" << min << std::endl;

    std::cout << "0~19分人数：" << counts[0] << std::endl;
    std::cout << "20~39分人数：" << counts[1] << std::endl;
    std::cout << "40~59分人数：" << counts[2] << std::endl;
    std::cout << "60~79分人数：" << counts[3] << std::endl;
    std::cout << "80~100分人数：" << counts[4] << std::endl;
}

int main() {
    int scores[MAX_STUDENTS];
    int n;
    int counts[5] = { 0 };
    int max = -1, min = -1;
    double avg = 0.0;
    int testIndex;

    // 1. 初始化数组
    initScores(scores, MAX_STUDENTS);

    // 2. 读取学⽣⼈数
    std::cout << "请输⼊学⽣⼈数：";
    std::cin >> n;

    // 3. 调⽤各函数（要求 0 <= score <= 100）
    inputScores(scores, n);

    avg = calcAverage(scores, n);

    findMinMax(scores, n, &max, &min);

    countGrades(scores, n, counts);

    // 7. 安全查询测试
    std::cout << "请输⼊要查询的下标：";
    std::cin >> testIndex;
    int queried = safeGet(scores, n, testIndex);
    if (queried == -1) {
        std::cout << "查询失败" << std::endl;
    } else {
        std::cout << "下标 " << testIndex << " 的成绩是：" << queried << std::endl;
    }

    // 8. 打印所有统计结果
    printResults(scores, n, avg, counts, max, min);
    return 0;
}