https://www.databricks.com/glossary/data-vault

- data vault 有三种不同的实体：hubs、links、satellites；
- business keys (Hubs), relationships (Links), and descriptive data (Satellites)
- Hubs 用于表示核心的商业概念
	- Each hub represents a core business concept, such as they represent Customer Id/Product Number/Vehicle identification number (VIN). Users will use a business key to get information about a Hub. The business key may have a combination of business concept ID and sequence ID, load date, and other metadata information.
- Links 表示 hubs 之间的关系；
- Satellites fill the gap in answering the missing descriptive information on core business concepts. Satellites store information that belongs to Hub and relationships between them.


![[Pasted image 20231016122558.png]]


