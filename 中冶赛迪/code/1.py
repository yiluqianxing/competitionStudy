#coding=utf-8
# import sys
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = 0
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = list(map(int, line.split()))
#         for v in values:
#             ans += v
#     print(ans)
# import sys
# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))
import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    # print("n",n)
    ans = 0
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        # print("ddd",values[0])
        ou = 0
        ji = 0
        num = 0
        j = 1
        while(j < values[0]):
            j = j + 2
            if values[0] % j == 0:
                tmpou = int(values[0]/j)
                tmpji = j
                if num ==0:
                    ou = tmpou
                    ji = tmpji
                    num = 32

                    continue
                if tmpou < ou:
                    ou = tmpou
                    ji = tmpji

        if ou != 0 and ji != 0:
            print(ji," ",ou)
        if ji == values[0]:
            print("No")