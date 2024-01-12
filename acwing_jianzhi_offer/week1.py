'''
13. 找出数组中重复的数字
https://www.acwing.com/problem/content/14/
给定一个长度为 n的整数数组 nums，数组中所有的数字都在 0∼n−1的范围内。
数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。
请找出数组中任意一个重复的数字。
注意：如果某些数字不在 0∼n−1
 的范围内，或数组中不包含重复数字，则返回 -1；
数据范围
0≤n≤1000
样例
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。
返回 2 或 3。
'''

class Solution(object):
    def duplicateInArray(self, nums):
        """
        :type nums: List[int]
        :rtype int
        """
        # 如果某些数字不在 0∼n−1的范围内，或数组中不包含重复数字，则返回 -1；
        len_num = len(nums)
        if not nums or max(nums) >= len_num or min(nums) < 0: return -1
        # 方法一：利用字典：时间复杂度：O(N)，空间复杂度：O(N)
        num_dict = {}
        for value in nums:
            if value not in num_dict:
                num_dict[value] = True
            else: return value
        return -1

        # 方法2：在原地重排序，把每个数放到对应位置上,对nums进行遍历，索引是idx，值是value
        # 如果value！= idx,就交换value和nums[value]的值，如果nums[value]!=nums[nums[value]]的值，就继续交换
        # 直到如果nums[idx] != idx,说明出现重复的，return nums[idx]
        for idx in range(len_num):
            while nums[idx] != nums[nums[idx]]:
                value = nums[idx]
                nums[idx], nums[value] = nums[value],nums[idx]
            if nums[idx] != idx: return nums[idx]
        return -1


"""
14. 不修改数组找出重复的数字
给定一个长度为 n+1的数组nums，数组中所有的数均在 1∼n的范围内，其中 n≥1
请找出数组中任意一个重复的数，但不能修改输入的数组。
数据范围
1≤n≤1000
样例
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。
返回 2 或 3。
"""


class Solution(object):
    def duplicateInArray(self, nums):
        """
        :type nums: List[int]
        :rtype int
        """
        # 方法1：用dict进行数据保存,如果不存在这个数就保存进去，如果存在这个数就说明重复，返回这个数
        # 时间复杂度O(N),空间复杂度O(N)
        # num_dict = {}
        # for value in nums:
        #     if value not in num_dict:
        #         num_dict[value] = True
        #     else: return value

        # 方法2：利用抽屉原理和分治思想：有n个箱子，放n+1个苹果，必然至少有一个箱子有两个苹果
        #        于是对箱子进行区间划分，比如6个箱子，我取前3个为左区间，4到6箱子为右区间
        #        对nums中的所有数进行遍历，数小于等于3，就把苹果以放进前三个箱子，其他的放进4~6箱子
        #        如果前3个箱子中一共放进去了4个苹果（有4个数小于等于3），说明重复数就在这边（左边），
        #        如果在大于3的箱子中，苹果数量大于箱子个数，那就是重复数在大于3的箱子中，
        #        进而用二分思想，继续进行区间分割，找出装有两个苹果的箱子
        # 时间复杂度O(NlogN),空间复杂度O(1)
        l, r = 1, len(nums) - 1
        while l < r:
            l_num = 0
            mid = int(l + r >> 1)
            for num in nums: l_num += num <= mid
            if l_num > mid:
                r = mid
            else:
                l = mid + 1
        return l

"""
15. 二维数组中的查找
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
数据范围
二维数组中元素个数范围 [0,1000]
样例
输入数组：
[
  [1,2,8,9]，
  [2,4,9,12]，
  [4,7,10,13]，
  [6,8,11,15]
]
如果输入查找数值为7，则返回true，
如果输入查找数值为5，则返回false。
"""
class Solution(object):
    def searchArray(self, array, target):
        """
        :type array: List[List[int]]
        :type target: int
        :rtype: bool
        取每行的最后一个数进行比较 ，如果最后一个数等于目标-->返回true；
                                如果小于目标，说明这一行都小，去掉
                                如果大于目标，说明这一行都大，去掉
        """
        if not target: return False
        n, m = len(array)-1, len(array[0])-1
        x, y = 0, m
        while x<=n and y>=0:
            if array[x][y] == target: return True
            if array[x][y] < target: # 整行不要
                x += 1
            else: y -= 1
        return False

"""
16. 替换空格
请实现一个函数，把字符串中的每个空格替换成"%20"。
数据范围
0≤输入字符串的长度 ≤1000。
注意输出字符串的长度可能大于 1000。
样例
输入："We are happy."
输出："We%20are%20happy."
"""
class Solution(object):
    def replaceSpaces(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s.replace(' ','%20')

"""
17. 从尾到头打印链表
输入一个链表的头结点，按照 从尾到头 的顺序返回节点的值。返回的结果用数组存储。
数据范围
0≤链表长度 ≤1000。
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def printListReversingly(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        res = []
        while head:
            res.append(head.val)
            head = head.next
        res.reverse()
        return res

"""
18. 重建二叉树
输入一棵二叉树前序遍历和中序遍历的结果，请重建该二叉树。
注意:
二叉树中每个节点的值都互不相同；
输入的前序遍历和中序遍历一定合法；
数据范围
树中节点数量范围 [0,100]。
样例
给定：
前序遍历是：[3, 9, 20, 15, 7]
中序遍历是：[9, 3, 15, 20, 7]
返回：[3, 9, 20, null, null, 15, 7, null, null, null, null]
返回的二叉树如下所示：
    3
   / \
  9  20
    /  \
   15   7
"""
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        前序遍历开头的数据，作为根节点root，找出root在中序遍历中的位置，此时左边是root的中序遍历左子树，右边是root中序遍历的右子树
        左边长度为l，则前序遍历中的root后面就切分出l长度的数据，作为左子树数据的构建，剩下的就是右子树的构建数据
        """
        index_map = {value:idx for idx,value in enumerate(inorder)}
        if not preorder: return None
        root = TreeNode(preorder[0])
        index = index_map[preorder[0]]
        root.left = self.buildTree(preorder[1:index+1],inorder[:index])
        root.right = self.buildTree(preorder[index+1:],inorder[index+1:])
        return root

"""
19. 二叉树的下一个节点
给定一棵二叉树的其中一个节点，请找出中序遍历序列的下一个节点。
注意：
如果给定的节点是中序遍历序列的最后一个，则返回空节点;
二叉树一定不为空，且给定的节点一定不是空节点；
数据范围
树中节点数量 [0,100]。
样例
假定二叉树是：[2, 1, 3, null, null, null, null]， 给出的是值等于2的节点。
则应返回值等于3的节点。
解释：该二叉树的结构如下，2的后继节点是3。
  2
 / \
1   3
"""
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.father = None
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.father = None
class Solution(object):
    def inorderSuccessor(self, q):
        """
        :type q: TreeNode
        :rtype: TreeNode
        """
        if q.right:
            q = q.right
            while q.left: q = q.left
            return q
        if q.father and q.father.left == q: return q.father
        while q.father and q.father.right==q: q = q.father
        return q.father

"""
20. 用两个栈实现队列
请用栈实现一个队列，支持如下四种操作：
push(x) – 将元素x插到队尾；
pop() – 将队首的元素弹出，并返回该元素；
peek() – 返回队首元素；
empty() – 返回队列是否为空；
注意：
你只能使用栈的标准操作：push to top，peek/pop from top, size 和 is empty；
如果你选择的编程语言没有栈的标准库，你可以使用list或者deque等模拟栈的操作；
输入数据保证合法，例如，在队列为空时，不会进行pop或者peek等操作；
数据范围
每组数据操作命令数量 [0,100]。
样例
MyQueue queue = new MyQueue();
queue.push(1);
queue.push(2);
queue.peek();  // returns 1
queue.pop();   // returns 1
queue.empty(); // returns false
"""


class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = []
        self.p = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        new = [x]
        new.extend(self.q)
        self.q = new

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.q.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.q[-1]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return not self.q

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()


"""
21. 斐波那契数列
"""
class Solution(object):
    def Fibonacci(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1: return n
        numlist = [0,1]
        for i in range(2,n+1):
            numlist.append(numlist[i-1]+numlist[i-2])
        return numlist[-1]


"""
22. 旋转数组的最小数字
例如数组 {3,4,5,1,2}
 为 {1,2,3,4,5}的一个旋转，该数组的最小值为 1。数组可能包含重复项。
注意：数组内所含元素非负，若数组大小为 0，请返回 −1。
数据范围
数组长度 [0,90]。
样例
输入：nums = [2, 2, 2, 0, 1]
输出：0
"""


class Solution:
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        {3,4,4,4,5,1,2}  {5,2,2,2,2,2,3} 把两边相等的数据去掉，，剩下的进行二分法
        """
        if not nums: return -1
        l, r = 0, len(nums) - 1
        while l < r and nums[l] == nums[l + 1]: l += 1
        while l < r and nums[r] == nums[r - 1]: r -= 1
        if l == r: return nums[l]

        while l < r and nums[l] >= nums[r]:
            mid = (l + r) >> 1  # [:mid]  [mid:]
            if nums[l] <= nums[mid]:
                l = mid + 1
            else:
                r = mid
        return nums[l]

"""
23. 矩阵中的路径
请设计一个函数，用来判断在一个矩阵中是否存在一条路径包含的字符按访问顺序连在一起恰好为给定字符串。
路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。
注意：
输入的路径字符串不为空；
所有出现的字符均为大写英文字母；
数据范围
矩阵中元素的总个数 [0,900]。
路径字符串的总长度 [1,900]。
matrix=
[
  ["A","B","C","E"],
  ["S","F","C","S"],
  ["A","D","E","E"]
]
str="BCCE" , return "true" 
str="ASAE" , return "false"
"""
class Solution(object):
    def hasPath(self, matrix, string):
        """
        :type matrix: List[List[str]]
        :type string: str
        :rtype: bool
        """
        if not matrix: return False
        self.n, self.m = len(matrix), len(matrix[0])
        self.dx, self.dy = [-1, 0, 1, 0], [0, -1, 0, 1]
        for x in range(self.n):
            for y in range(self.m):
                if self.dfs(x, y, string, matrix): return True

        return False

    def dfs(self, x, y, string, matrix):
        if not string: return True
        if 0 <= x < self.n and 0 <= y < self.m and matrix[x][y] and matrix[x][y] == string[0]:
            s = matrix[x][y]
            matrix[x][y] = False
            for i in range(4):
                if self.dfs(x + self.dx[i], y + self.dy[i], string[1:], matrix): return True
            matrix[x][y] = s

        return False