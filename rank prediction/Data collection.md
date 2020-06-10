# Data Collection

http://ultimatetennisstatistics.com is known for his complete statistics on tennis you can almost find everything, you would look for to predict tennis outcomes.

Mcekovic is Serbian and did an amazing work in gathering information about tennis, he scrapped a lot of data from https://atptour.com, we thank him for this because without him this project would be a lot more complicated than it is already.
Here is a link to his github where you can have more detail on his project :
https://github.com/mcekovic/tennis-crystal-ball

Recently, he also opened an issue on his github where he put a link to his Docker container that can be pulled. It contains a replica of his database with tennis data until the end of the 2019 season.

We will use this database for the data collection part.

Let's choose the features we will extract from the database.

We will take matches from beginning of 2014 to end of 2017 for training.

After we will validate our model on 2018 data.

After we may study ROI on 2019 data.

Features of match table :
- match_id
- date
- winner_id
- loser_id
- winner_rank
- loser_rank
- winner_rank_points
- loser_rank_points
- winner_elo_rating
- loser_elo_rating

We have the following SQL query for training data :
```sql
SELECT match_id, date, winner_id, loser_id, winner_rank, loser_rank, winner_rank_points, loser_rank_points, winner_elo_rating, loser_elo_rating FROM match WHERE date >= '2014-01-01' AND date <= '2017-12-31'
```

Which give the following command to extract in csv :
```bash
docker exec -it uts-database psql -U tcb -d tcb -c "COPY (SELECT match_id, date, winner_id, loser_id, winner_rank, loser_rank, winner_rank_points, loser_rank_points, winner_elo_rating, loser_elo_rating FROM match WHERE date >= '2014-01-01' AND date <= '2017-12-31') TO STDOUT WITH CSV HEADER " > "2014-2017.csv"
```

For 2018 data we have :
```bash
docker exec -it uts-database psql -U tcb -d tcb -c "COPY (SELECT match_id, date, winner_id, loser_id, winner_rank, loser_rank, winner_rank_points, loser_rank_points, winner_elo_rating, loser_elo_rating FROM match WHERE date >= '2018-01-01' AND date <= '2018-12-31') TO STDOUT WITH CSV HEADER " > "2018.csv"
```

For 2019 data we have :
```bash
docker exec -it uts-database psql -U tcb -d tcb -c "COPY (SELECT match_id, date, winner_id, loser_id, winner_rank, loser_rank, winner_rank_points, loser_rank_points, winner_elo_rating, loser_elo_rating FROM match WHERE date >= '2019-01-01' AND date <= '2019-12-31') TO STDOUT WITH CSV HEADER " > "2019.csv"
```