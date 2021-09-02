# Quiz3: Using MYSQL, Analysis of DB

# 1. 멕시코(Mexico)보다인구가많은나라이름과인구수를조회하시고인구수순으로내림차순하세요. 

```
select name, population
from country
where population > (select population from country where code = "MEX")
order by population desc;
```

# 2. 국가별몇개의도시가있는지조회하고도시수순으로 10위까지내림차순하세요.

```
select country.name, count(city.name) as count
from country
join city
on country.code = city.countrycode
group by country.name
order by count desc
limit 10;
```

# 3. 언어별사용인구를출력하고언어사용인구순으로 10위까지내림차순하세요.

```
select language, round(sum(country.population*countrylanguage.percentage/100), 0) as popltn
from country
join countrylanguage
on country.code = countrylanguage.countrycode
group by language
order by popltn desc
limit 10;
```

# 4. 나라전체인구의 10%이상인도시에서도시인구가 500만이넘는도시를아래와같이조회하세요.

```
select city.name, country.code, country.name, round(city.population/country.population*100, 2) as percentage 
from country
join city
on country.code = city.countrycode
where city.population >= (country.population * 0.1)
and city.population > 500*10000
order by percentage desc;
```
-----
```
select city_name, country_code, country_name, percentage
from
	(
	select city.name as city_name 
		   , country.code as country_code
           , country.name as country_name
           , city.population as city_popltn
           , round(city.population/country.population*100, 2) as percentage 
	from country
	join city
	on country.code = city.countrycode
	where city.population >= (country.population * 0.1)
	) A
where city_popltn > 500*10000
order by percentage desc;
```

# 5. 면적이 10000km^2 이상인국가의인구밀도(1km^2 당인구수)를구하고인구밀도(density)가 200이상인국가들의사용하고있는언어가 2가지인나라를조회하세요. 출력 : 국가이름, 인구밀도, 언어수출력

```
select name, density, language_count, language_list
from (select code, name, population, SurfaceArea, round(population/SurfaceArea, 0) as density 
	  from country 
      where SurfaceArea >= 10000 
	  having density > 200) A
join (select countrycode, group_concat(language) as language_list, count(language) as language_count 
	  from countrylanguage 
      group by countrycode) B
on A.code = B.countrycode
having language_count = 2;
```

# 6. 사용하는언어가 3가지이하인국가중도시인구가 300만이상인도시를아래와같이조회하세요.

```
select C3ll.countrycode, city.name as city_name, city.Population
	   , C3ll.name, C3ll.language_count, C3ll.languages   
from city
join C3ll
on city.countrycode = C3ll.countrycode
having city.population >= 300*10000
order by city.population desc;
```

```
create view C3ll as
select CountryCode, country.name
	   , group_concat(language) as languages
	   , count(language) as language_count
from countrylanguage
join country
on countrylanguage.countrycode = country.code
group by countrycode
having language_count <= 3;
```

# Quiz 7. 한국와미국의인구와 GNP를세로로아래와같이나타내세요. (쿼리문에국가코드명을문자열로사용해도됩니다.)

```
select
	Max(case when category = 'KOR' then round(A.population,0) end) as 'KOR',
    Max(case when category = 'USA' then round(A.population,0) end) as 'USA'
from 
	(
	select code as category, population, gnp
	from country
	where code in ("KOR", "USA")
	) A
union
select
	Max(case when category = 'KOR' then round(A.gnp,0) end) as 'KOR',
    Max(case when category = 'USA' then round(A.gnp,0) end) as 'USA'
from 
	(
	select code as category, population, gnp
	from country
	where code in ("KOR", "USA")
	) A
;
```

# Quiz 8. sakila 데이터베이스의 payment 테이블에서수입(amount)의총합을아래와같이출력하세요.


```
select
	MAX(case when Ym = '2005-05' then A.tot_amount end) as '2005-05',
    sum(case when Ym = '2005-06' then A.tot_amount end) as '2005-06',
	sum(case when Ym = '2005-07' then A.tot_amount end) as '2005-07',
    sum(case when Ym = '2005-08' then A.tot_amount end) as '2005-08',
    sum(case when Ym = '2006-02' then A.tot_amount end) as '2006-02'
from 
(
select date_format(payment_date, "%Y-%m") as Ym, sum(amount) as tot_amount 
from payment
group by Ym
) A;
```

# Quiz 9. 위의결과에서 payment 테이블에서월별렌트횟수데이터를추가하여아래와같이출력하세요.

```
select
	MAX(case when Ym = '2005-05' then A.tot_amount end) as '2005-05',
    sum(case when Ym = '2005-06' then A.tot_amount end) as '2005-06',
	sum(case when Ym = '2005-07' then A.tot_amount end) as '2005-07',
    sum(case when Ym = '2005-08' then A.tot_amount end) as '2005-08',
    sum(case when Ym = '2006-02' then A.tot_amount end) as '2006-02'
from 
(
select date_format(payment_date, "%Y-%m") as Ym, sum(amount) as tot_amount 
from payment
group by Ym
) A
union
select
	MAX(case when Ym = '2005-05' then A.tot_rent end) as '2005-05',
    sum(case when Ym = '2005-06' then A.tot_rent end) as '2005-06',
	sum(case when Ym = '2005-07' then A.tot_rent end) as '2005-07',
    sum(case when Ym = '2005-08' then A.tot_rent end) as '2005-08',
    sum(case when Ym = '2006-02' then A.tot_rent end) as '2006-02'
from 
(
select date_format(payment_date, "%Y-%m") as Ym, count(amount) as tot_rent 
from payment
group by Ym
) A;
```
