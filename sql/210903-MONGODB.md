## Done

	- Replication  
		: feature to synchronize a lot of DB servers

	- tmux
	  : For two or more coworkers, they can enter same server

	- MONGODB : NOSQL
		: DB that don't have relationship with tables
		
		- No foriegn key & join
		- table > collection
		- row, record > document
		- slow select(read), but fast insert(create)

			-> So, when using collect big data
	
		: mongodb is adapted to Javascript
			
			- Data Base Produce
				
				```use Mongo```

			- show selected DB now
				
				```db```

   			- show database list

				```show dbs```

			- Make new Document

				```
		    db.user.insert({"name": "peter", "age": 30})
		    		```
			- Creat Collection
			- db.createCollection(name, [option])
			- option 
		: capped, autoIndex, size(byte), max(document) 
			
			-- limit of capacity 
			
			-- rolling concept on saving mode

			```db.createCollection("user")```

		- CRUD : CREATE
```db.createCollection("info1", {autoIndexId: true, capped: true, size: 500, max: 5})
db.createCollection("info2", {autoIndexId: true, capped: true, size: 50, max: 5})

db.article.insert({"title": "ssac", "content": "mongodb"})

show collections```

		- Delete Collection
		
			```
		db.article.drop()
			```

		- Add Document in empty Collection
```
db.info1.insert({"name": "peter", "age": 26})
db.info1.insert({"name": "kevin", "age": 25})
db.info1.insert({"name": "max", "age": 22})

db.info1.insert([
    {"name": "ruby", "age": 2},
    {"name": "max", "age": 4},
    {"name": "max", "age": 25},
])

db.info2.insert([
    {"name": "ruby", "age": 2},
    {"name": "max", "age": 4},
    {"name": "max", "age": 25},
    {"name": "ma", "age": 23},
    {"name": "m", "age": 21},
])
```

		### CRUD : READ : find
		- find(query, projection)
```db.info1.find()

db.info1.find({age: 2})
```
		- comparison operator
		: $lt <, $lte <=, %gt >, $gte >=, $eq =
			```
		db.info1.find({age: {$gt: 2}})
			```
		- $in
			```
		db.info1.find({name: {$in: ["java", "css"]})
			```
		- logical operator
		: $or, $and, $not, $nor

```
db.info1.find({$and: [{name: "kevin"}, {age: {$gte: 2}}]})
```
			-- check "{"!! + $and, $not, $nor, $or

			-- always check database "open shell"

		- projection
			```
	db.info1.find({}, {_id: false, "age": false})
			```
		- sort : 1:asc, -1:desc
			```
		db.info1.find().sort({age: 1})
		db.info1.find().sort({age: -1})
			```
			-- ".function" chaining

		- limit
			```
		db.info1.find().limit(3).sort({age: 1})
			```
		- skip (mysql limit 3, 5: skip 3 and print 5)
			```
		db.info1.find().skip(3).limit(5)
			```
		### CRUD: UPDATE

		- update(Query, set, option)
			```
		db.info1.update(
    			{name: "kevin"},
    			{name: "jaechan", age: 89}
		)
			```
			-- we have to input all category data
```
db.info1.update(
    {name: "max"},
    {name: "max", age: 46},
    {upsert: true}
)
```
			-- if there is no jaechan, append new row
		- $set
```		
db.info1.update(
    {name: "ruby"},
    {$set: {age: 89}}
)
```

		- multi: change a lot of data simultaneously
```
db.info1.update(
    {name: "max"},
    {$set: {age: 49}},
    {multi: true}
)
```

		- function: bind code as specific feature.
```
var pageNation = function(page, pageBlock){
    return db.info1.find().skip((page-1)*pageBlock).limit(pageBlock)
}
pageNation(2,3)
```

## To Do

- Python Basic Programming(1)
