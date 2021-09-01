## Quiz 2: Using MYSQL, Analysis of DB

# Quiz 1. 국가코드별도시의갯수를출력하세요. (상위 5개를출력)
select countrycode, count(name) as count
from city
group by CountryCode
order by count desc
limit 5;

# Quiz 2. 대륙별몇개의국가가있는지대륙별국가의갯수로내림차순하여출력하세요.
select continent, count(name) as count
from country
group by continent
order by count desc;

# Quiz 3. 대륙별인구가 1000만명이상인국가의수와 GNP의평균을소수둘째자리에서반올림하여첫째자리까지출력하세요.
select continent, count(name), round(avg(gnp), 1) as avg_gnp
from country
where population >= 1000*10000
group by continent
order by avg_gnp desc;

# Quiz 4. city 테이블에서국가코드(CountryCode) 별로총인구가몇명인지조회하고총인구순으로내림차순하세요. (총인구가 5천만이상인도시만출력)
select countrycode, sum(population) as population
from city
group by countrycode
having population >= 5000*10000
order by population desc;

# Quiz 5. countrylanguage 테이블에서언어별사용하는국가수를조회하고많이사용하는언어를 6위에서 10위까지조회하세요.
select Language, count(CountryCode) as count
from countrylanguage
group by Language
order by count desc
limit 5, 5;

# Quiz 6. countrylanguage 테이블에서언어별 20개국가이상에서사용되는언어를조회하고언어별사용되는국가수에따라내림차순하세요.
select Language, count(CountryCode) as count
from countrylanguage
group by Language
having count >= 20
order by count desc;

# Quiz 7. country 테이블에서대륙별전체표면적크기를구하고표면적크기순으로내림차순하세요.
select continent, sum(surfacearea) as SurfaceArea
from country
group by continent
order by SurfaceArea desc;

# Quiz 8. World 데이터베이스의 countrylanguage에서언어의사용비율이 90%대(90 ~ 99.9)의사용율을갖는언어의갯수를출력하세요.
select count(distinct(language)) as count_90
from countrylanguage
where percentage >= 90;

# Quiz 9. 1800년대에독립한국가의수와 1900년대에독립한국가의수를출력하세요.
select A.indepyear_ages, count(A.code) as count
from
	(
		select Indepyear,
				  case
						when IndepYear between 1900 and 1999 then 1900
						when IndepYear between 1800 and 1899 then 1800
				   end as indepyear_ages, code
		from country
	) A
GROUP BY A.indepyear_ages
ORDER BY A.indepyear_ages desc
limit 2;

# Quiz 10. sakila의 payment 테이블에서월별총수입을출력하세요.
use sakila;

select monthly, sum(A.amount)
from
	(
	select date_format(payment_date,"%Y-%m") as monthly, amount
	from payment
	order by monthly asc
	) A
group by monthly
order by monthly asc;

# Quiz 11. actor 테이블에서가장많이사용된first_name을아래와같이출력하세요.

select A.first_name
from
	(
	select first_name, count(first_name) as count
	from actor
	group by first_name
	order by count desc
	) A
where A.count = 4;

# Quiz 12. film_list 뷰에서카테고리별가장많은매출을올린카테고리 3개를매출순으로정렬하여아래와같이출력하세요.

select A.category
from
	(
	select category, sum(price) as Price
	from film_list
	group by category
	order by Price desc
	limit 3
	) A;

