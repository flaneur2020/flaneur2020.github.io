允许客户部署在他们自己的 VPC 中。

> These challenges included setting up and managing the VPC, ensuring network connectivity, implementing auto-provisioning mechanisms for cloud resources, and providing the right level of abstraction to make the solution both user-friendly and flexible.

## Auto-provisioning of Cloud Resources

### Easy onboarding

用了 CloudFormation 和 Crossplane 来自动化这些资源的管理。

customer 只需要关注：

1. 创建 IAM Role：使用 CloudFormation 提供的模板来创建 role
2. Configuration and Provisioning：指定 region、VPC CIDR range、AZ，创建相关的 cloud 资源；
3. 在 cloud resources 创建完成之后，客户可以启动自己的 clickhouse service；
### Cloud infrastructure separation

设计思想：拆分 clickhouse 的管理服务，和 EKS cluster 中的数据面。

![[Screenshot 2025-03-14 at 19.23.23.png]]

- **Management Services** 还在 clickhouse 的 vpc 中，不能访问 customer data；
- **Data plane**：在客户的 VPC 中，The data plane is customizable, allowing the customer to configure EC2 instances, VPC settings, and security groups to meet their specific needs.

## Data isolation and compliance

logs 和 metrics 总在客户自己的 vpc 中；

data transfer 会有限制，只有推送用量、telemetry 数据会传送。

### Access for troubleshooting

- Access must be approved and audited.
- Access permissions must be separated for each BYOC setup.
- Permissions must be limited and controlled. For example, reading secrets and executing into a pod will not be allowed.

Our solution follows a controlled escalation process:

1. Engineers request access through an internal approval system.
2. A designated approvers group reviews and grants access if necessary.
3. The system temporarily enables access for the approved engineer, which automatically expires after a set time.

会有一个内部审核系统，通过审核之后，工程师可以临时获得访问权限。

#### Certificate based auth for system table access

支持工程师会需要连接到 clickhouse 的 system table 上。

这时会通过 tailscale 连过去，使用一个临时证书进行认证。

1. On-call engineers within the designated Okta group make a request to access the customer ClickHouse instance system table. This also generates a time bound certificate.
2. The ClickHouse operator configures ClickHouse to accept the certificate.
3. On-call engineers access the instance via Tailscale using the certificate.

BYOC 环境中，所有的 support 都是基于证书的。

会阻止所有的 password 访问。

### Scheduled upgrades

We give BYOC customers full control over upgrades by allowing them to define a [scheduled upgrade time window](https://clickhouse.com/docs/en/manage/updates#scheduled-upgrades). Customers can specify a preferred maintenance window to ensure that updates happen at a time that minimizes impact on their workloads.