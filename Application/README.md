# Install and Run Application
## Requirements
Docker and docker-compose

## Build and Run
1. docker-compose up

However the application requiers login, so to create a user run the command below.
Here you will be promted for username, email and password.

2. docker-compose run web python Arx/manage.py createsuperuser
3. Run docker compose-up
4. go to: http://0.0.0.0:8000

The docker build will fill the database with the two csv files found in
the Data folder

how to delete docker images
https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes

# Further Development
## Backend
Run docker-compose up and start programming.

## Frontend
It is the backend that hosts the react frontend. So to further develop the frontend run the command
npm run dev, when you have changed some code, and place the main.js file insid Arx/frontend/static/frontend
and then run docker-compose up.
