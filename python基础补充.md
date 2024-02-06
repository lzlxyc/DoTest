解决git中不能push的问题
https://blog.csdn.net/qq_46123200/article/details/134392486


# 一、内置队列
## 1、LILO队列
- 先进先出
```python
from queue import Queue
# 创建队列,实力化队列类 
q = Queue()
# 向队尾添加元素
q.put()
# 查看队列中的所有元素
q.queue
# 进一步的，获取队列中的第几个元素
q.queue[0],q.queue[-1]
# 返回对头的元素，并删除
q.get()
# 查看队列是否为空
q.empty()

from queue import LifoQueue  # 使用方法同上
```
## 2、优先队列
- 指定优先级进行元素的添加，后续元素弹出时就会按照优先级的排序,通常采用堆数据结构来实现,并进行优先级的维护
- 在往队列中加入元素的时候第一个元素值表示的是元素的优先级，并且值越小那么优先级越高；
- 所以队首元素的优先级是最高的，而且经常加入队列的元素类型为元组这样就可以在队列中保存多个值
```python
from queue import PriorityQueue
p = PriorityQueue()
p.put((7,'a'))
p.put((-1,'b'))
p.put((100,'c'))
p.queue # 此时输出[(7,'a'),(-1,'b'),(100,'c')]
# 在进行元素输出时，按照优先级
p.get() # 此时输出（-1，'b')
p.get() # 此时输出（7,'a')
p.get() # 此时输出（100，'c')
```

## 3、双端队列
```python
from collections import deque
d = deque([1,2,3,4])
# 在右边添加
d.append()
# 在左边添加
d.appendleft()
# 返回并删除右边元素
d.pop()
# 返回并删除左边元素
d.popleft()

```

## itertools.permutations


## cmp_to_key比较数值大小
```python
from functools import cmp_to_key
def cmp(a,b):
    return a - b 
l = [(2,1),(1,4),(6,4)]
l.sort(key=cmp_to_key(cmp))
```

