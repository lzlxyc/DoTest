"""
24. 机器人的运动范围
地上有一个 m行和 n列的方格，横纵坐标范围分别是 0∼m−1和 0∼n−1。
一个机器人从坐标 (0,0)的格子开始移动，每一次只能向左，右，上，下四个方向移动一格。
但是不能进入行坐标和列坐标的数位之和大于 k的格子。
请问该机器人能够达到多少个格子？
注意:
0<=m<=50
0<=n<=50
0<=k<=100
样例1
输入：k=7, m=4, n=5
输出：20
"""


class Solution(object):
    def movingCount(self, threshold, rows, cols):
        """
        :type threshold: int
        :type rows: int
        :type cols: int
        :rtype: int
        宽度优先搜索题目，将满足条件的点添加进list中，每次取出一个，又将它周围满足的点添加进list，
        对所有次数进行累加，得到最终结果
        """
        if not rows * cols: return 0
        mm = [[False] * cols for _ in range(rows)]
        self.k, res = threshold, 0
        dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
        q = [(0, 0)]
        while q:
            x, y = q.pop()
            if mm[x][y] or self.over(x, y): continue
            res += 1
            mm[x][y] = True
            for i in range(4):
                a, b = x + dx[i], y + dy[i]
                if 0 <= a < rows and 0 <= b < cols: q.append((a, b))

        return res

    def over(self, x, y):
        return self.k < sum([int(num) for num in list(str(x))]) + sum([int(num) for num in list(str(y))])

"""
25. 剪绳子
给你一根长度为 n绳子，请把绳子剪成 m段（m、n
都是整数，2≤n≤58并且 m≥2）。
每段的绳子的长度记为 k[1]、k[2]、……、k[m]。
k[1]k[2]…k[m]可能的最大乘积是多少？
例如当绳子的长度是 8时，我们把它剪成长度分别为 2、3、3的三段，此时得到最大的乘积 18。
样例
输入：8
输出：18
"""
class Solution(object):
    def maxProductAfterCutting(self,length):
        """
        :type length: int
        :rtype: int
        """
        num_map = {0:3,1:4,2:6}
        if length <=3: return length-1
        last = length % 3
        num = int(length/3)
        '''
        if last == 0: return 3**num
        if last == 1: return 3**(num-1)*4
        if last == 2: return 3**(num)*2
        '''
        return 3**(num-1)*num_map[last]

"""
26. 二进制中1的个数
输入一个 32位整数，输出该数二进制表示中 1的个数。
注意：
负数在计算机中用其绝对值的补码来表示。数据范围−100≤输入整数 ≤100
样例1
输入：9
输出：2
解释：9的二进制表示是1001，一共有2个1。
"""
class Solution(object):
    def NumberOf1(self,n):
        """
        :type n: int
        :rtype: int
        """
        res, idx = int(n < 0), 32
        if n < 0: n = 2**31 + n
        while n:
            idx -= 1
            num = 2**idx
            if num > n: continue
            else:
                res += 1
                n -= num
        return res

"""
27. 数值的整数次方
实现函数double Power(double base, int exponent)，求base的 exponent次方。
不得使用库函数，同时不需要考虑大数问题。只要输出结果与答案的绝对误差不超过 10−2即视为正确。
注意：
不会出现底数和指数同为0的情况当底数为0时，指数一定为正底数的绝对值不超过 10，指数的取值范围 [−231,231−1]。
样例1
输入：10 ，2
输出：100
"""


class Solution(object):
    def Power(self, base, exponent):
        """
        :type base: float
        :type exponent: int
        :rtype: float
        """
        if not base * exponent: return 0 if not base else 1
        '''
        方法1
        res = 1
        if exponent < 0:
            base = 1/base
            exponent = -exponent
        g = 1 if base > 0 or not exponent%2 else -1
        base = abs(base)
        while exponent: 
                res *= base
                exponent -= 1
                if res < 0.02:
                    return 0
        return g*res
        '''
        # 方法2：快速幂
        res = 1
        g = int(exponent < 0)
        k = abs(exponent)
        while k:
            if k & 1: res *= base
            base *= base
            k >>= 1
        if g: res = 1 / res
        return res


"""
28. 在O(1)时间删除链表结点
给定单向链表的一个节点指针，定义一个函数在O(1)时间删除该结点。
假设链表一定存在，并且该节点一定不是尾节点。
数据范围
链表长度 [1,500]。
样例
输入：链表 1->4->6->8删掉节点：第2个节点即6（头节点为第0个节点）
输出：新链表 1->4->8
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void
        将下一个节点的值赋值给当前节点，将next指向下一个的下一个，即把自己替换成next，删除原本的next
        """
        p = node.next
        node.val = node.next.val
        node.next = node.next.next
        del p

"""
29. 删除链表中重复的节点
在一个排序的链表中，存在重复的节点，请删除该链表中重复的节点，重复的节点不保留。
数据范围
链表中节点 val 值取值范围 [0,100]。
链表长度 [0,100]。
样例1
输入：1->2->3->3->4->4->5
输出：1->2->5
"""
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def deleteDuplication(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next: return head
        s = p = ListNode(-1)
        s.next, p.next = head, head
        while head and head.next:
            if head.val != head.next.val:
                p = head
                head = head.next
            else:
                while head and head.next and head.val==head.next.val:
                    head = head.next
                head = head.next
                p.next = head
        return s.next


"""hard
30. 正则表达式匹配(https://www.acwing.com/problem/content/28/)
请实现一个函数用来匹配包括'.'和'*'的正则表达式。
模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。
在本题中，匹配是指字符串的所有字符匹配整个模式。
例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配。
数据范围
输入字符串长度 [0,300]。
样例
输入：
s="aa"
p="a*"
输出:true
"""


"""hard
31. 表示数值的字符串(https://www.acwing.com/problem/content/29/)
实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。
但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
注意:
小数可以没有整数部分，例如.123等于0.123；
小数点后面可以没有数字，例如233.等于233.0；
小数点前面和后面可以有数字，例如233.666;
当e或E前面没有数字时，整个字符串不能表示数字，例如.e1、e1；
当e或E后面没有整数时，整个字符串不能表示数字，例如12e、12e+5.4;
数据范围
输入字符串长度 [0,25]。
字符串中不含空格。
样例：
输入: "0"
输出: true
"""


"""
32. 调整数组顺序使奇数位于偶数前面
输入一个整数数组，实现一个函数来调整该数组中数字的顺序。
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分。
数据范围
数组长度 [0,100]。
数组内元素取值范围 [0,100]。
样例
输入：[1,2,3,4,5]
输出: [1,3,5,2,4]
"""
class Solution(object):
    def reOrderArray(self, array):
        """
        :type array: List[int]
        :rtype: void
        """
        if not array: return []
        l, r = 0, len(array) - 1
        while l < r:
            while l < r and array[l] % 2: l += 1
            while l < r and array[r] % 2 == 0: r -= 1
            array[l], array[r] = array[r], array[l]
        return array

"""
33. 链表中倒数第k个节点
输入一个链表，输出该链表中倒数第 k个结点。
注意：
k >= 1;如果 k大于链表长度，则返回 NULL;
数据范围
链表长度 [0,30]。
样例
输入：链表：1->2->3->4->5 ，k=2
输出：4
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def findKthToTail(self, pListHead, k):
        """
        :type pListHead: ListNode
        :type k: int
        :rtype: ListNode
        """
        len_node, p = 0, pListHead
        while p:
            len_node += 1
            p = p.next
        if len_node < k: return None
        len_node -= k
        p = pListHead
        while len_node:
            p = p.next
            len_node -= 1
        return p

"""图
34. 链表中环的入口结点(https://www.acwing.com/problem/content/86/)
给定一个链表，若其中包含环，则输出环的入口节点。
若其中不包含环，则输出null。
数据范围
节点 val 值取值范围 [1,1000]。
节点 val 值各不相同。
链表长度 [0,500]。
"""
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def entryNodeOfLoop(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''方法1，用字典做判断，时间O(n),空间O(n)
        num_dict = {}
        p = head
        while p:
            if p.val not in num_dict:
                num_dict[p.val] = True
            else: return p
            p = p.next
        return None
        '''
        '''
        方法2，用快慢指针时间O(n),空间O(1)
        第一步：快指针2倍速行走，慢指针一倍速，这样如果有环，就一定会相遇
        第二步：如果相遇，就在相遇的点，让快指针从头开始，慢指针变成一倍速，再次相遇的点就是环的入口点

        '''
        p = q = head  # p慢，q快
        while q and p:
            p = p.next
            q = q.next
            if not q or not q.next:
                return None
            else:
                q = q.next
            if p == q:
                p = head
                while p != q:
                    q = q.next
                    p = p.next
                return p
        return None







