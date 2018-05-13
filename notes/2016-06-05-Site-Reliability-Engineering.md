---
layout: default
title: Site Reliability Engineering
---

# 读书笔记: Site Reliability Engineering


## The Virtue of Boring 

<原文开始>Essential complexity is the complexity inherent in a given situation that cannot be removed from a problem definition, whereas accidental complexity is more fluid and can be resolved with engineering effort</原文结束>
## Gmail: Predictable, Scriptable Response from Humans

<原文开始>This kind of tension is common within a team, and often reflects an underlying mistrust of the team's discipline: while some team members want to implement a "hack" to allow time for a proper fix, others worry that a hack will be forgotten or that the proper fix will be deprioritized indefinitely. This concern is credible, as it's easy to build layers of unmaintainable technical debt by patching over problems instead of making real fixes. Managers and technical leaders play a key role in implementing true, long-term fixes by supporting and prioritizing potentially time-consuming long-term fixes even when the initial "pain" of paging subsides. </原文结束>
## Cost

<原文开始>One useful strategy may be consider the background error rate of ISPs on the Internet. If failures are being measured from the end-user perspective and it is possible to drive the error rate for the service below the background error rate, those errors will fall within the noise for the given user's internet connection.</原文结束>

服务的 SLA 不比运营商差就好 
## Other service metrics

<原文开始>In other words, given the relative insensitivity of the AdSense service to moderate changes in latency performance, we are able to consolidate serving into fewer geographical locations, reducing our operational overhead.
 </原文结束>

如果延时不那么敏感，那么可以省几个异地机房
## Forewood

<原文开始>Nothing here tells us how to solve problems universally, but that is the point. Stories like these are far more valuable than the code or designs they resulted in. Implementations are ephemeral, but the documented reasoning is priceless. Rarely do we have access to this kind of insights. </原文结束>