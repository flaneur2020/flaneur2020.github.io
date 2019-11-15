package logmerger

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

type LogConsumerStats struct {
	MessagesTotal        uint64 `json:"messagesTotal"`
	PreparesTotal        uint64 `json:"preparesTotal"`
	CommitsTotal         uint64 `json:"commitsTotal"`
	LastMessageTimestamp int64  `json:"lastMessageTimestamp"`
}

var (
	token int64
	step  int64

	maxSleepInterval int64 = 5
	maxGap           int64 = 10
)

// LogConsumer 接口对接消息源，如 Kafka、Binlog 等。
// LogConsumer 遇到任何错误皆应从上次点位开始继续消费
type LogConsumer interface {
	Run(poistion int64, quitc chan struct{})
	Close()
	Iterator() LogIterator
	Stats() LogConsumerStats
}

type FakeLogConsumer struct {
	inputc  chan Log
	outputc chan Log

	producer *FakeProducer

	sleepInterval time.Duration
	step          int64
	stats         LogConsumerStats
}

var _ LogConsumer = &FakeLogConsumer{}

func NewFakeLogConsumer(inputc chan Log) *FakeLogConsumer {
	return &FakeLogConsumer{
		inputc:  inputc,
		outputc: make(chan Log, 1024),

		sleepInterval: 1 * time.Second,
		stats:         LogConsumerStats{},
	}
}

func (c *FakeLogConsumer) Run(position int64, quitc chan struct{}) {
	for {
		select {
		case <-quitc:
			return
		case lg := <-c.inputc:
			c.stats.LastMessageTimestamp = time.Now().Unix()
			c.stats.MessagesTotal++
			if lg.kind == "commit" {
				c.stats.CommitsTotal++
			} else if lg.kind == "prepare" {
				c.stats.PreparesTotal++
			}
			c.output(lg, quitc)
		}
	}
}

func (c *FakeLogConsumer) output(lg Log, quitc chan struct{}) {
	select {
	case <-quitc:
	case c.outputc <- lg:
	}
}

func (c *FakeLogConsumer) Close() {
	close(c.outputc)
}

func (c *FakeLogConsumer) Stats() LogConsumerStats {
	return c.stats
}

func (c *FakeLogConsumer) Iterator() LogIterator {
	var it LogIterator
	it = NewChanIterator(c.outputc)
	it = NewCommitedLogIterator(it)
	// the step is expected to be multiple of 5 * 10
	return NewStagedIterator(it, 100)
}

type FakeProducer struct {
	logc chan Log
}

func NewFakeProducer() *FakeProducer {
	return &FakeProducer{logc: make(chan Log, 20)}
}

func (p *FakeProducer) Output() chan Log {
	return p.logc
}

func (p *FakeProducer) Produce() {
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		p.generateDatas()
	}()
	go func() {
		defer wg.Done()
		p.generateDatas()
	}()
	wg.Wait()
}

func (p *FakeProducer) generateDatas() {
	for {
		prepare := incrementToken()
		sleep(maxSleepInterval)

		p.logc <- Log{kind: "prepare", prepareToken: prepare}
		sleep(maxSleepInterval)

		commit := incrementToken()
		sleep(maxSleepInterval)

		p.logc <- Log{kind: "commit", prepareToken: prepare, commitToken: commit}
		sleep(10 * maxSleepInterval)
	}
}

func incrementToken() int64 {
	return atomic.AddInt64(&token, rand.Int63()%maxGap+1)
}

func sleep(factor int64) {
	// 最短睡眠时间：1ms
	// 最长睡眠时间: factor * 1ms
	interval := atomic.AddInt64(&step, 3)%factor + 1
	waitTime := time.Duration(rand.Int63() % interval)
	time.Sleep(waitTime * time.Millisecond)
}
