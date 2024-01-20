"""
35. 反转链表
定义一个函数，输入一个链表的头结点，反转该链表并输出反转后链表的头结点。
思考题：
请同时实现迭代版本和递归版本。
数据范围
链表长度 [0,30]。
样例
输入:1->2->3->4->5->NULL
输出:5->4->3->2->1->NULL
"""
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''方法1：用迭代法
        if not head or not head.next: return head
        l, m, r = head, head.next,head.next.next
        m.next = l 
        while r:
            l, m, r = m, r, r.next
            m.next = l 
        head.next = None
        return m
        '''
        '''方法2：用递归法'''
        if not head or not head.next: return head
        tail = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return tail

"""
36. 合并两个排序的链表
输入两个递增排序的链表，合并这两个链表并使新链表中的结点仍然是按照递增排序的。
数据范围
链表长度 [0,500]。
样例
输入：1->3->5 , 2->4->5
输出：1->2->3->4->5->5
"""
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def merge(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        p = head = ListNode(-1)
        while l1 and l2:
            if l1.val < l2.val:
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        if l1: p.next = l1
        if l2: p.next = l2
        return head.next

"""
37. 树的子结构
输入两棵二叉树 A，B，判断 B是不是 A的子结构。
我们规定空树不是任何树的子结构。
数据范围
每棵树的节点数量 [0,1000]。
样例
树 A：
     8
    / \
   8   7
  / \
 9   2
    / \
   4   7
树 B：
   8
  / \
 9   2
返回 true
"""
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def hasSubtree(self, pRoot1, pRoot2):
        """
        :type pRoot1: TreeNode
        :type pRoot2: TreeNode
        :rtype: bool
        1)遍历所有A的点，都作为A根节点
        2）输入A的根节点，B的根节点，进行递归，看看是否构成子树：如果A和B不相等或者A没有了B还有，说明不构成子树
        3）递归到B没有了，说明构成了子树
        """
        if not pRoot2 or not pRoot1: return False
        if self.dfs(pRoot1, pRoot2): return True
        return self.hasSubtree(pRoot1.left, pRoot2) or self.hasSubtree(pRoot1.right, pRoot2)

    def dfs(self, r1, r2):
        if not r2: return True
        if not r1 or r1.val != r2.val: return False
        return self.dfs(r1.left, r2.left) and self.dfs(r1.right, r2.right)

"""
38. 二叉树的镜像
输入一个二叉树，将它变换为它的镜像。
数据范围
树中节点数量 [0,100]。
样例
输入树：
      8
     / \
    6  10
   / \ / \
  5  7 9 11
 [8,6,10,5,7,9,11,null,null,null,null,null,null,null,null] 
输出树：
      8
     / \
    10  6
   / \ / \
  11 9 7  5
 [8,10,6,11,9,7,5,null,null,null,null,null,null,null,null]
"""
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def mirror(self, root):
        """
        :type root: TreeNode
        :rtype: void
        """
        if not root: return
        root.left, root.right = root.right, root.left
        self.mirror(root.left)
        self.mirror(root.right)

"""
39. 对称的二叉树
请实现一个函数，用来判断一棵二叉树是不是对称的。
如果一棵二叉树和它的镜像一样，那么它是对称的。
数据范围
树中节点数量 [0,100]。
样例
如下图所示二叉树[1,2,2,3,4,4,3,null,null,null,null,null,null,null,null]为对称二叉树：
    1
   / \
  2   2
 / \ / \
3  4 4  3
如下图所示二叉树[1,2,2,null,4,4,3,null,null,null,null,null,null]不是对称二叉树：
    1
   / \
  2   2
   \ / \
   4 4  3
"""
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.dfs(root, root)

    def dfs(self, rleft, rright):
        if not rleft and not rright: return True
        if not rleft and rright or not rright and rleft or rleft.val != rright.val: return False
        # 剩下的就是有left又有right
        return self.dfs(rleft.right, rright.left) and self.dfs(rleft.left, rright.right)

"""
40. 顺时针打印矩阵
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
数据范围
矩阵中元素数量 [0,400]。
样例
输入：
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
"""
class Solution(object):
    def printMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix or not matrix[0]: return []
        n, m = len(matrix), len(matrix[0])
        st = [[False]*m for _ in range(n)]
        x = y = 0
        dx, dy = (1,0,-1,0), (0,1,0,-1)
        idx, res = 0, []
        tot = n*m
        for _ in range(tot):
            res.append(matrix[y][x])
            st[y][x] = True
            a,b = x + dx[idx], y + dy[idx]
            if a < 0 or a >= m or b < 0 or b >= n or st[b][a]:
                idx = (idx+1)%4
                a,b = x + dx[idx], y + dy[idx]
            x, y = a, b
        return res

"""
41. 包含min函数的栈
设计一个支持push，pop，top等操作并且可以在O(1)时间内检索出最小元素的堆栈。 
push(x)–将元素x插入栈中
pop()–移除栈顶元素
top()–得到栈顶元素
getMin()–得到栈中最小元素
"""


class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.q = []
        self.m = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.q.append(x)
        if not self.m or x <= self.m[-1]: self.m.append(x)

    def pop(self):
        """
        :rtype: void
        """
        if self.q[-1] == self.m[-1]: self.m.pop()
        self.q.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.q[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.m[-1]

"""
42. 栈的压入、弹出序列
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
假设压入栈的所有数字均不相等。
例如序列 1,2,3,4,5是某栈的压入顺序，序列 4,5,3,2,1是该压栈序列对应的一个弹出序列，但 4,3,5,1,2就不可能是该压栈序列的弹出序列。
注意：若两个序列长度不等则视为并不是一个栈的压入、弹出序列。若两个序列都为空，则视为是一个栈的压入、弹出序列。
数据范围
序列长度 [0,1000]。
样例
输入：[1,2,3,4,5]
      [4,5,3,2,1]
输出：true
"""


class Solution(object):
    def isPopOrder(self, pushV, popV):
        """
        :type pushV: list[int]
        :type popV: list[int]
        :rtype: bool
        """
        len_in, len_out = len(pushV), len(popV)
        if len_in != len_out: return False

        tmp = []
        k = 0
        for i in range(len_in):
            tmp.append(pushV[i])
            while tmp and tmp[-1] == popV[k]:
                tmp.pop()
                k += 1
        return not tmp

"""
43. 不分行从上往下打印二叉树
从上往下打印出二叉树的每个结点，同一层的结点按照从左到右的顺序打印。
数据范围
树中节点的数量 [0,1000]。
样例
输入如下图所示二叉树[8, 12, 2, null, null, 6, null, 4, null, null, null]
    8
   / \
  12  2
     /
    6
   /
  4
输出：[8, 12, 2, 6, 4]
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from queue import Queue


class Solution:
    def printFromTopToBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        tmp = Queue()
        tmp.put(root)
        res = []
        while not tmp.empty():
            p = tmp.get()
            res.append(p.val)
            if p.left: tmp.put(p.left)
            if p.right: tmp.put(p.right)
        return res

"""
44. 分行从上往下打印二叉树
从上到下按层打印二叉树，同一层的结点按从左到右的顺序打印，每一层打印到一行。
数据范围
树中节点的数量 [0,1000]。
样例
输入如下图所示二叉树[8, 12, 2, null, null, 6, null, 4, null, null, null]
    8
   / \
  12  2
     /
    6
   /
  4
输出：[[8], [12, 2], [6], [4]]
"""
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from queue import Queue
class Solution(object):
    def printFromTopToBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        res, res_tmp = [], []
        q = Queue()
        q.put(root), q.put(None)
        while not q.empty():
            tmp = q.get()
            if tmp:
                res_tmp.append(tmp.val)
                if tmp.left: q.put(tmp.left)
                if tmp.right: q.put(tmp.right)
            else:
                res.append(res_tmp)
                res_tmp = []
                if not q.empty(): q.put(None)
        return res

"""
45. 之字形打印二叉树
请实现一个函数按照之字形顺序从上向下打印二叉树。
即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
数据范围
树中节点的数量 [0,1000]。
样例
输入如下图所示二叉树[8, 12, 2, null, null, 6, 4, null, null, null, null]
    8
   / \
  12  2
     / \
    6   4
输出：[[8], [2, 12], [6, 4]]
"""
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

from queue import Queue
class Solution(object):
    def printFromTopToBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        res,res_tmp = [], []
        q = Queue()
        q.put(root), q.put(None)
        flag = False
        while not q.empty():
            tmp = q.get()
            if tmp:
                res_tmp.append(tmp.val)
                if tmp.left: q.put(tmp.left)
                if tmp.right: q.put(tmp.right)
            else:
                if flag: res_tmp.reverse()
                flag = not flag
                res.append(res_tmp)
                res_tmp = []
                if not q.empty(): q.put(None)
        return res