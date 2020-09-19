package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func bubbleSortList(head *ListNode) *ListNode {
	for p := head; p != nil; p = p.Next {
		for q := head; q != nil; q = q.Next {
			if p.Val < q.Val {
				tmp := p.Val
				p.Val = q.Val
				q.Val = tmp
			}
		}
	}
	return head
}

func quickSortList(head *ListNode, tail *ListNode) {
	if head == tail || head.Next == tail {
		return
	}

	pp := head
	for p := head.Next; p != tail; p = p.Next {
		if p.Val < pp.Val {
			tmp := p.Val
			p.Val = pp.Val
			pp.Val = tmp
			pp = p
		}
	}

	quickSortList(head, pp)
	quickSortList(pp.Next, nil)
}

func main() {
	l := &ListNode{Val: 4}
	l.Next = &ListNode{Val: 2}
	l.Next.Next = &ListNode{Val: 1}
	l.Next.Next.Next = &ListNode{Val: 3}

	quickSortList(l, nil)
	fmt.Printf("hello: %d\n", l.Val)
	fmt.Printf("hello: %d\n", l.Next.Val)
	fmt.Printf("hello: %d\n", l.Next.Next.Val)
	fmt.Printf("hello: %d\n", l.Next.Next.Next.Val)
}
