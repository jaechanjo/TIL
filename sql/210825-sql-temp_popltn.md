## Done

- Establish environment of SQL

	- Download Oracle DB/ DB Tool: DBeaver
	- Make user account/ Table Space: DASQL
	- Connect account to Oracle

- Analysis of Temperature

	- Data Import
	- SELECT () FROM ()
	- ORDER BY ASC/ DESC
	- WHERE () = (): Condition about specific spot, feature
	- SELF JOIN: Stragedy to analyse DB by setting a table two or more object
	- Inlinen View: In From syntax, one more SELECT-FROM Syntax.
	- ROWNUM: turn about output values
	- NULL: NULL is most upper than any number/ always NULL when including NULL
	- SUBSTR('text',1,2): substract without range first to second character
	- Division of row: CASE WHEN () BETWEEN A AND B AS/THEN C

- Analysis of Population

	- Dupulication technique being advantage of "Connect By" with "Dual Table" of Oracle
		: it' like binding books
		```SELECT LEVEL AS () FROM DUAL CONNECT BY LEVEL <= (num)```

		Then, Why did we use this technique?
		: In order to convert complex classification with apparent that, multiply the sources.

	- Analysis Function/ Widow Func.
		```RATIO_TO_REPORT/ROW_NUMBER/ROUND/SUM () OVER (PARTITION () ORDER BY () ASC/DESC```
		
	- NOT LIKE: Strategy Of Complementary Set
		```WHERE () NOT LIKE '______00000'```
	
	- Strategy seleting 1st
		```ROW_NUMBER () OVER () AS A(turn) ... WHERE A(turn) = 1
	
## TO DO

- Analysis of Public Transportation

