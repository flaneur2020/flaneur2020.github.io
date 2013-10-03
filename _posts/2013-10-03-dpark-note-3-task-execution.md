---
layout: post
title: "Dpark Note 3: Task Execution"
---

在前面一些预备内容之后，任务执行的悬念其实已经不大了。DAGScheduler 会在 runJob 中多次调用 submitTasks，由调度器后端负责具体的执行，随后调度器后端调用 taskEnded 向 DAGScheduler 反馈任务的进展。

## LocalScheduler

先看最简单的 LocalScheduler，它将任务安排在本地单线程执行，代码非常直接，几乎没有注释的必要。 它代表了调度器后端的最简实现。

```
def run_task(task, aid):
    logger.debug("Running task %r", task)
    try:
        Accumulator.clear()
        result = task.run(aid)
        accumUpdates = Accumulator.values()
        return (task.id, Success(), result, accumUpdates)
    except Exception, e:
        logger.error("error in task %s", task)
        import traceback
        traceback.print_exc()
        return (task.id, OtherFailure("exception:" + str(e)), None, None)

class LocalScheduler(DAGScheduler):
    attemptId = 0
    def nextAttempId(self):
        self.attemptId += 1
        return self.attemptId

    def submitTasks(self, tasks):
        logger.debug("submit tasks %s in LocalScheduler", tasks)
        for task in tasks:
            _, reason, result, update = run_task(task, self.nextAttempId())
            self.taskEnded(task, reason, result, update)
```

## MesosScheduler

MesosScheduler 同时扮演着 dpark 调度器的后端和 mesos 调度器的客户端，负责集群执行的相关逻辑。其中包含了许多字段，按功能分类主要有：

* 记录向 Mesos 申请 Offer 的要求，比如 cpus, mem, task_per_node, group，可见 dpark 在一次集群计算过程中对于所有 Task 都是同样的资源要求；
* Mesos 相关的工具对象，比如 driver, executor；
* 维护 SimpleJob 的生命周期及相关的薄记信息，如 activeJobs, activeJobsQueue, taskIdToJobId, taskIdToSlaveId，这几个对象的用法跟 DAGScheduler 中维护 Stage 生命周期的做法几乎来自同一个模子，只是维护的是另一层对象的生命周期；
* 保存日志收集线程的端点地址：output_logger, error_logger。在初始化 MesosScheduler 时会开两个守护线程侦听来自各节点的日志，这里用到了 zeromq 的 PULL 模式；
* 健康信息：last_finished_time，记录上次任务完成的时间。在两个收集日志的守护线程之外，MesosScheduler 还会开一个守护线程作看门狗，如果过长时间没有新任务完成，则自杀并报错；
* 锁: lock，确保同一时刻只有一个线程拥有 MesosDriver 对象的访问权；

MesosScheduler 算是 dpark 与 mesos 交集最大的地方：既针对 DAGScheduler 实现了 submitTasks() 方法，也针对 mesos.Scheduler 实现了一系列事件响应方法。在初始化之后，计算的入口就是 submitTasks() 方法了，从它开始看：

```
@safe
def submitTasks(self, tasks):
    if not tasks:
        return

    job = SimpleJob(self, tasks, self.cpus, tasks[0].rdd.mem or self.mem)
    self.activeJobs[job.id] = job
    self.activeJobsQueue.append(job)
    self.jobTasks[job.id] = set()
    logger.info("Got job %d with %d tasks: %s", job.id, len(tasks), tasks[0].rdd)

    need_revive = self.started
    if not self.started:
        self.start_driver()
    while not self.isRegistered:
        self.lock.release()
        time.sleep(0.01)
        self.lock.acquire()

    if need_revive:
        self.requestMoreResources()
```

submitTasks() 方法首先创建了一个 SimpleJob 对象，用以维护 Mesos 任务的生命周期。SimpleJob 维护任务生命周期的做法仍与前面 DAGScheduler 维护 Stage 生命周期的做法很相似。留意 SimpleJob 其中 "Job" 的含义与 Scheduler#runJob() 中的 "Job" 的含义大相径庭 ，在逻辑上 Jobs > Stages =~ SimpleJobs > Tasks ，Stage 和 SimpleJob 的执行是平行的：

* Job 有多个 Stage，生命周期由 DAGScheduler 管理，根据后端反馈的任务执行情况调整 Stage 的执行状态；保证只有已满足全部依赖的任务才会执行，待所有任务执行完毕整个 Stage 才算执行完毕；
* 来自 Stage 的所有任务将全数包装成 SimpleJob 交给 Mesos，它们包含的任务基本相同，但侧重点差异很大： SimpleJob 是完全与 Mesos 相关的结构，与 Stage 没有耦合，也并不关心任务之间的依赖；它仅仅负责把任务搭配 Offer 包装成 Mesos 接纳的格式发送给 Mesos，并跟踪 Mesos 任务的执行状态，以及在任务级别维护必要的重试。从 Mesos 得到确认所有任务执行完毕之后，SimpleJob 算执行完毕。

可见一个 Mesos 任务在发送之前，需要经过 DAGScheduler, MesosScheduler, SimpleJob 三层中转。

与 Mesos 交互的流程在笔记一曾简单带过：在发送任务之前需要先向 Mesos 申请 Offer；随后收到 Offer，会触发 MesosScheduler#resourceOffers() 方法，在这里真正向 Mesos 提交任务。它会遍历 activeJobsQueue，按顺序为 SimpleJob 对象选出合适的 Offer，将其中所有的 Task 对象序列化，搭配 Offer 申明的资源包装成一个 Protobuf 格式的 Mesos Task，最终调用 driver.launchTasks 发送给 Mesos。

任务具体的部署就属于 Mesos 的工作了，它会将序列化后的 Task 对象传递到 Offer 申明的计算节点，最终交给部署在上面的 Executor 进程。

## MyExecutor

MyExecutor 中执行任务的 run_task 函数实现其实与 LocalScheduler 的 run_task 函数大同小异，不同在于解序列化、保存计算结果到本地以及向 Mesos 返回任务的执行结果：

```
def run_task(task_data):
    try:
        gc.disable()
        task, ntry = cPickle.loads(decompress(task_data))
        setproctitle('dpark worker %s: run task %s' % (Script, task))

        Accumulator.clear()
        result = task.run(ntry)
        accUpdate = Accumulator.values()

        if marshalable(result):
            try:
                flag, data = 0, marshal.dumps(result)
            except Exception, e:
                flag, data = 1, cPickle.dumps(result, -1)

        else:
            flag, data = 1, cPickle.dumps(result, -1)
        data = compress(data)

        if len(data) > TASK_RESULT_LIMIT:
            path = LocalFileShuffle.getOutputFile(0, ntry, task.id, len(data))
            f = open(path, 'w')
            f.write(data)
            f.close()
            data = '/'.join([LocalFileShuffle.getServerUri()] + path.split('/')[-3:])
            flag += 2

        return mesos_pb2.TASK_FINISHED, cPickle.dumps((Success(), (flag, data), accUpdate), -1)
    except FetchFailed, e:
        return mesos_pb2.TASK_FAILED, cPickle.dumps((e, None, None), -1)
    except :
        import traceback
        msg = traceback.format_exc()
        return mesos_pb2.TASK_FAILED, cPickle.dumps((OtherFailure(msg), None, None), -1)
    finally:
        setproctitle('dpark worker: idle')
        gc.collect()
        gc.enable()
```

此外，MyExecutor 作为守护进程还有其它事情需要做：

* 建立 HttpServer 保存计算结果，允许其它节点抓取；
* 转发日志：开单独的线程侦听 stdin 的输出，通过 zeromq 的 PUSH 模式发送到 Master;
* 开单独的线程作看门狗，通过 self.check_memory，检查 worker 进程的健康状态，比如是否僵死，是否内存占用过多;

## Summary

阅读 dpark 的代码时感叹了好多次：它总比我想象中的更为复杂。可见代码的精简并非意味着容易理解，在关键的抽象之外，细节往往更值得花费更多注意力。与代码相比，文字的描述终究难逃成为一个偷工减料的过程。比如 Shuffle 的细节，和 broadcast 的实现，在这里都被偷工减料掉了。它们其实都很有趣，被省略的原因只是懒散的人难以驾驭更长的文章，请读者留意。
