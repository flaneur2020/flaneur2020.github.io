package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func popList(l *ListNode) (*ListNode, *ListNode) {
	if l == nil {
		return nil, nil
	}

	nl := l.Next
	l.Next = nil
	return nl, l
}

func reverseLinkedList(l *ListNode, k int) (*ListNode, *ListNode) {
	var (
		n, h *ListNode
	)
	l, h = popList(l)
	if l == nil {
		return h, nil
	}
	for l != nil && k > 1 {
		l, n = popList(l)
		n.Next = h
		h = n
		k--
	}
	return h, l
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	var tail *ListNode
	lastp := head
	head, tail = reverseLinkedList(head, k)
	finalHead := head
	for {
		lastp.Next = tail
		lastp = tail
		_, tail = reverseLinkedList(tail, k)
		if tail == nil {
			break
		}
	}
	return finalHead
}

func main() {
	l := &ListNode{Val: 1}
	l.Next = &ListNode{Val: 2}
	l.Next.Next = &ListNode{Val: 3}
	l.Next.Next.Next = &ListNode{Val: 4}
	l.Next.Next.Next.Next = &ListNode{Val: 5}
	l.Next.Next.Next.Next.Next = &ListNode{Val: 6}
	lastp := l
	head, tail := reverseLinkedList(l, 3)
	lastp.Next = tail
	lastp = tail
	_, tail = reverseLinkedList(tail, 3)
	lastp.Next = tail
	fmt.Printf(
		"%v %v %v %v %v %v",
		head.Val,
		head.Next.Val,
		head.Next.Next.Val,
		head.Next.Next.Next.Val,
		head.Next.Next.Next.Next.Val,
	)
}
