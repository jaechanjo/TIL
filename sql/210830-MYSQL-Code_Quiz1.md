## Using population & film data

### 한국 도시 중 인구 수가 100만이 넘는 도시를 인구수 순으로 오름차순 정렬 출력하세요.

select name, population
from city
where (countrycode = "KOR") and (population >= 100*10000)
order by population desc;

### 도시인구가 800만 ~1000만 사이인 도시의 데이터를 국가 코드 순 오름차순 출력하세요.

select countrycode, name, population
from city
where population between 800*10000 and 1000*10000
order by countrycode asc;

### 1940 ~ 1950년도사이에독립한국가들중 GNP가 10만이넘는국가를 GNP의내림차순으로출력하세요.

select code, name, continent, gnp
from country
where IndepYear Between 1940 and 1950 and gnp >= 10*10000
order by gnp desc;

#  스페인어(Spanish), 영어(English), 한국어(Korean) 중에 95% 이상사용하는국가코드, 언어, 비율을출력하세요.

select countrycode, language, percentage
from countrylanguage
where language in ('Spanish', 'English', 'Korean') and percentage >= 95
order by percentage desc;

# 국가코드가 "K"로시작하는국가중에기대수명(lifeexpectancy)이 70세이상인국가를기대수명의내림차순순으로출력하세요. 

select code, name, continent, LifeExpectancy
from country
where code like "K%" and LifeExpectancy >= 70
order by LifeExpectancy desc;

use sakila;

#film_text 테이블에서 title이 ICE가들어가고 description에 Drama가들어간데이터를출력하세요.

select *
from film_text
where title like "%ICE%" and description like "%Drama%";

# actor 테이블에서이름(first_name)의가장앞글자가 "A", 성(last_name)의가장마지막글자가 "N"으로끝나는배우의데이터를출력하세요.

select actor_id, first_name, last_name
from actor
where first_name like "A%" and last_name like "%N"; 

# film 테이블에서 rating이 "R" 등급인film 데이터를상영시간(length)이가장긴상위 10개의film을상영시간의내림차순순으로출력하세요.

select film_id, title, description, rental_duration, rental_rate, length, rating
from film
where rating = "R"
order by length desc
limit 0, 10;

# 상영시간(length)이 60분 ~ 120분인필름데이터에서영화설명(description)에robot 들어있는영화를상영시간(length)이짧은순으로오름차순하여정렬하고, 11위에서 13위까지의영화를출력하세요.

select * -- film_id, title, description, length
from film;
where length between 60 and 120 and description like "%robot%"
order by length asc
limit 10, 3;

# film_list view에서카테고리(category)가 sci-fi, anmation, drama가아니고배우(actors) 가 "ed chase", "kevin bloom" 이포함된영화리스트에서상영시간이긴순서대로 5개의영화리스트를출력하세요. 

select title, description, category, length, actors
from film_list
where category not in ('sci-fi','animation','drama') and actors like "%ed chase%" or "%kevin bloom%"
order by length desc
limit 0, 5;
