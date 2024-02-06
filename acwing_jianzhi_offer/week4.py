"""
46. 二叉搜索树的后序遍历序列
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则返回true，否则返回false。
假设输入的数组的任意两个数字都互不相同。
数据范围
数组长度 [0,1000]。
样例
输入：[4, 8, 6, 12, 16, 14, 10]
输出：true
"""
class Solution:
    def verifySequenceOfBST(self, sequence):
        """
        :type sequence: List[int]
        :rtype: bool
        """
        len_tree = len(sequence)
        if len_tree <= 1: return True
        root = sequence[-1]
        left,right = [], []
        k = 0
        while k<len_tree-1 and sequence[k] < root:
            left.append(sequence[k])
            k += 1
        while k < len_tree-1:
            if sequence[k] <= root: return False
            right.append(sequence[k])
            k += 1
        return self.verifySequenceOfBST(left) and self.verifySequenceOfBST(right)

"""
47. 二叉树中和为某一值的路径
输入一棵二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
保证树中结点值均不小于 。
数据范围
树中结点的数量 [0,1000]。
样例
给出二叉树如下所示，并给出num=22。
      5
     / \
    4   6
   /   / \
  12  13  6
 /  \    / \
9    1  5   1
输出：[[5,4,12,1],[5,6,6,5]]
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findPath(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        self.res, self.tmp = [], []
        self.dfs(root, sum)
        return self.res

    def dfs(self, p, sum):
        if not p: return
        self.tmp.append(p.val)
        sum -= p.val
        if sum == 0 and not p.left and not p.right:  # 达到条件
            self.res.append([r for r in self.tmp])
        if p.left: self.dfs(p.left, sum)
        if p.right: self.dfs(p.right, sum)
        self.tmp.pop()
        return

"""
48. 复杂链表的复刻
请实现一个函数可以复制一个复杂链表。
在复杂链表中，每个结点除了有一个指针指向下一个结点外，还有一个额外的指针指向链表中的任意结点或者null。
注意：
函数结束后原链表要与输入时保持一致。
数据范围
链表长度 [0,500]。
"""


# Definition for singly-linked list with a random pointer.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#         self.random = None
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        """
        方法1：时间O(N^2)，空间O(N)
        res = ListNode(-1)
        p,q = head, res

        while p:
            q.next = ListNode(p.val)
            q = q.next
            p = p.next
        p,q = head,res.next
        while p:
            r_head,r_res = head, res.next
            while r_head and p.random !=r_head:
                r_head = r_head.next
                r_res = r_res.next
            q.random = r_res
            p = p.next
            q = q.next
        return res.next
        """

        # 方法2：时间O（N），空间O(N)
        dict_note = {}
        res = ListNode(-1)
        p = res
        while head:
            if head not in dict_note: dict_note[head] = ListNode(head.val)
            if head.random and head.random not in dict_note: dict_note[head.random] = ListNode(head.random.val)
            p.next = dict_note[head]
            p.next.random = None if not head.random else dict_note[head.random]
            p = p.next
            head = head.next
        return res.next

"""
49. 二叉搜索树与双向链表(不会)
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。
注意：
需要返回双向链表最左侧的节点。
例如，输入下图中左边的二叉搜索树，则输出右边的排序双向链表。
"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def convert(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root: return root
        self.p = None
        self.dfs(root)
        while root and root.left: root = root.left
        return root

    def dfs(self, root):
        if not root: return
        if root.left:
            self.dfs(root.left)
        root.left = self.p
        if self.p: self.p.right = root
        self.p = root
        if root.right:
            self.dfs(root.right)
        return

"""https://www.acwing.com/problem/content/46/
50. 序列化二叉树(困难)
请实现两个函数，分别用来序列化和反序列化二叉树。
您需要确保二叉树可以序列化为字符串，并且可以将此字符串反序列化为原始树结构。
数据范围
树中节点数量 [0,1000]。
样例
你可以序列化如下的二叉树
    8
   / \
  12  2
     / \
    6   4
为："[8, 12, 2, null, null, 6, 4, null, null, null, null]"
"""


"""
51. 数字排列
输入一组数字（可能包含重复数字），输出其所有的排列方式。
数据范围
输入数组长度 [0,6]。
样例
输入：[1,2,3]
输出：
      [
        [1,2,3],
        [1,3,2],
        [2,1,3],
        [2,3,1],
        [3,1,2],
        [3,2,1]
      ]
"""


class Solution:
    def permutation(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        len_ = len(nums)
        if len_ <= 1: return [nums]
        nums.sort()
        self.res, self.tmp = [], []
        self.dfs(nums)
        return self.res

    def dfs(self, num_list):
        if len(set(num_list)) == 1:
            self.res.append([n for n in self.tmp] + num_list)
            return
        tmp = [n for n in num_list]
        for i, num in enumerate(tmp):
            if num == tmp[i - 1]: continue
            self.tmp.append(num)
            num_list.remove(num)
            self.dfs(num_list)
            self.tmp.pop()
            num_list = [n for n in tmp]
        return

"""
52. 数组中出现次数超过一半的数字
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
假设数组非空，并且一定存在满足条件的数字。
思考题：
假设要求只能使用 O(n)的时间和额外 O(1)的空间，该怎么做呢？
数据范围
数组长度 [1,1000]。
样例
输入：[1,2,1,1,3]
输出：1
"""
class Solution(object):
    def moreThanHalfNum_Solution(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        p, c = nums[0], 1
        len_ = len(nums)
        for i in range(1, len_):
            if nums[i] == p:
                c += 1
            else:
                c -= 1
            if c == 0: p = nums[i]
        return p

"""
53. 最小的k个数
输入 n个整数，找出其中最小的 k个数。
注意：
输出数组内元素请按从小到大顺序排序;
数据范围
1≤k≤n≤1000
样例
输入：[1,2,3,4,5,6,7,8] , k=4
输出：[1,2,3,4]
"""
class Solution(object):
    def getLeastNumbers_Solution(self, input, k):
        """
        :type input: list[int]
        :type k: int
        :rtype: list[int]
        """
        input.sort()
        return input[:k]


"""https://www.acwing.com/problem/content/88/
54. 数据流中的中位数(困难）
如何得到一个数据流中的中位数？
如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
数据范围
数据流中读入的数据总数 [1,1000]。
样例
输入：1, 2, 3, 4
输出：1,1.5,2,2.5
解释：每当数据流读入一个数据，就进行一次判断并输出当前的中位数。
"""

"""
55. 连续子数组的最大和
输入一个 非空 整型数组，数组里的数可能为正，也可能为负。
数组中一个或连续的多个整数组成一个子数组。
求所有子数组的和的最大值。
要求时间复杂度为 O(n)。
数据范围
数组长度 [1,1000]。
数组内元素取值范围 [−200,200]。
样例
输入：[1, -2, 3, 10, -4, 7, 2, -5]
输出：18
"""
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_ = max(nums)
        t = 0
        for i in range(len(nums)):
            t += nums[i]
            if nums[i] >0:
                max_ = max(max_,t)
            if t < 0: t = 0
        return max_


"""https://www.acwing.com/problem/content/51/
56. 从1到n整数中1出现的次数(困难）
输入一个整数 n，求从 1到 n这 n个整数的十进制表示中 1出现的次数。
例如输入 12，从 1到 12这些整数中包含 “1”的数字有 1，10，11和 12，其中 “1”一共出现了 5次。
数据范围
1≤n≤109
样例
输入： 12
输出： 5
"""



























