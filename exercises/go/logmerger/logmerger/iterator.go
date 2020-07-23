package logmerger

import (
	"container/heap"
	"time"
)

type LogIterator interface {
	Current() *Log
	Next() bool
}

type ChanIterator struct {
	logc    <-chan Log
	current *Log
}

func NewChanIterator(logc <-chan Log) *ChanIterator {
	return &ChanIterator{
		logc:    logc,
		current: nil,
	}
}

func (it *ChanIterator) Current() *Log {
	if it.current != nil {
		return it.current
	}
	if ok := it.Next(); !ok {
		return nil
	}
	return it.current
}

func (it *ChanIterator) Next() bool {
	select {
	case log, ok := <-it.logc:
		if !ok {
			// always set it.current as nil on returning false
			it.current = nil
			return false
		}
		it.current = &log
		return true
	}
}

// commitedLogIterator combines the prepare message and commit messages
type CommitedLogIterator struct {
	it      LogIterator
	current *Log
}

func NewCommitedLogIterator(it LogIterator) *CommitedLogIterator {
	return &CommitedLogIterator{
		it:      it,
		current: nil,
	}
}

func (it *CommitedLogIterator) Current() *Log {
	if it.current != nil {
		return it.current
	}
	if ok := it.Next(); !ok {
		return nil
	}
	return it.current
}

func (it *CommitedLogIterator) Next() bool {
	for {
		if ok := it.it.Next(); !ok {
			it.current = nil
			return false
		}

		log := it.it.Current()
		if log == nil {
			it.current = nil
			return false
		}

		if log.kind == "commit" {
			it.current = log
			return true
		}
	}
}

// MergedIterator 取多个 iterator 按外部排序合并。注意，保全序的要求下，
// 其中任一 Iterator 阻塞会使整个 mergeIterator 阻塞。 初期简单的处理
// 办法可以是:
// `1. 监控 Consumer 的状态，如果超时则触发报警， 记录当前点位，退出并重启
//  2. 使每个 Producer 产生心跳消息，避免单一来源长时间没有写入，卡住所有
type MergedIterator struct {
	h       iteratorsHeap
	current *Log
}

func NewMergedIterator(iterators []LogIterator) *MergedIterator {
	h := iteratorsHeap(iterators)
	heap.Init(&h)
	return &MergedIterator{
		h:       h,
		current: nil,
	}
}

func (mi *MergedIterator) Current() *Log {
	if mi.current != nil {
		return mi.current
	}
	if ok := mi.Next(); !ok {
		return nil
	}
	return mi.current
}

func (mi *MergedIterator) Next() bool {
	if len(mi.h) == 0 {
		mi.current = nil
		return false
	}
	it := heap.Pop(&mi.h).(LogIterator)
	mi.current = it.Current()
	if mi.current == nil {
		return false
	}
	if ok := it.Next(); ok {
		heap.Push(&mi.h, it)
	}
	return true
}

// StagedIterator 取单个 Iterator 对 N 次迭代做排序，每次迭代
// 返回 N 次之前的数据
type StagedIterator struct {
	h       logsHeap
	size    int
	it      LogIterator
	current *Log
}

func NewStagedIterator(it LogIterator, size int) *StagedIterator {
	h := logsHeap([]Log{})
	heap.Init(&h)
	return &StagedIterator{
		h:       h,
		current: nil,
		size:    size,
		it:      it,
	}
}

func (it *StagedIterator) Current() *Log {
	if it.current != nil {
		return it.current
	}
	if ok := it.Next(); !ok {
		return nil
	}
	return it.current
}

func (st *StagedIterator) Next() bool {
	// push until the heap is full
	for len(st.h) < st.size {
		if ok := st.it.Next(); !ok {
			return false
		}
		heap.Push(&st.h, st.it.Current())
	}
	lg := heap.Pop(&st.h).(*Log)
	st.current = &*lg
	return true
}

// ThrottledIterator 允许指定时间窗口与时间窗口内的迭代次数上限
type ThrottledIterator struct {
	it LogIterator

	threshold  uint64
	timeWindow time.Duration

	count    uint64
	lastTick time.Time
}

func NewThrottledIterator(it LogIterator, threshold uint64, timeWindow time.Duration) *ThrottledIterator {
	return &ThrottledIterator{
		it:         it,
		count:      0,
		lastTick:   time.Now(),
		threshold:  threshold,
		timeWindow: timeWindow,
	}
}

func (it *ThrottledIterator) Current() *Log {
	return it.it.Current()
}

// 每当计数抵达 threshold, 则比较上次的时间是否小于 timeWindow，如果是，则 sleep 到 timeWindow
func (it *ThrottledIterator) Next() bool {
	it.count++
	if it.count > it.threshold {
		currentTick := time.Now()
		duration := currentTick.Sub(it.lastTick)
		if duration <= it.timeWindow {
			time.Sleep(it.timeWindow - duration)
		}
		// reset the counter and tick
		it.count = 0
		it.lastTick = currentTick
	}
	return it.it.Next()
}
