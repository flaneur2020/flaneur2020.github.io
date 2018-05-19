---
layout: default
title: Akka in Action
---

# 读书笔记: Akka in Action

<https://book.douban.com/people/fleure/annotation/24528141/>
## Futures

先记一个坑...

<代码开始 lang="java">
class AccountBalanceRetriever(savingsAccounts: ActorRef,
                              checkingAccounts: ActorRef,
                              moneyMarketAccounts: ActorRef) extends Actor {
  implicit val timeout: Timeout = 100 milliseconds
  implicit val ec: ExecutionContext = context.dispatcher
  def receive = {
    case GetCustomerAccountBalances(id) =>
      val futSavings = savingsAccounts ? GetCustomerAccountBalances(id)
      val futChecking = checkingAccounts ? GetCustomerAccountBalances(id)
      val futMM = moneyMarketAccounts ? GetCustomerAccountBalances(id)
      val futBalances = for {
        savings <- futSavings.mapTo[Option[List[(Long, BigDecimal)]]]
        checking <- futChecking.mapTo[Option[List[(Long, BigDecimal)]]]
        mm <- futMM.mapTo[Option[List[(Long, BigDecimal)]]]
      } yield AccountBalances(savings, checking, mm)
      futBalances map (sender ! _)
  }
}
</代码结束>

代码在 《effective scala》里摘的... 作为一个 bug 的例子。

<原文开始>The sender could have a completely different value at the time the callback is invoked.</原文结束>

Future 的回调函数里的这个 sender 是一个变量而不是值，当回调函数触发时，sender 已经不是原来的 sender 了。

解法是使用 pipeTo 或者：

val responder = sender
futBalances.map(responder ! _)

简直跟 javascript 有一拼了，现在是多讨厌闭包。线上出一个这样的 bug 得怎么调试呢。