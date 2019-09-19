import React, { Component } from "react";
import PropTypes from "prop-types";
import { Link } from 'react-router-dom';

import Typography from '@material-ui/core/Typography';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';

import Header from "./../components/Header";

const styles = {
  card: {
    width: 240,
    margin: 10,
    float: "left",
  },
  media: {
    height: 140,
    width: 240,
  },
  title: {
    textAlign: "center",
    fontSize: 24,
  },
  outer: {
    position:'absolute',
    width: "100%",
  },
  inner: {
    margin: "0 Auto",
    height: 240,
    width: 270*3,
    position:'relative',
  },
};

class Home extends Component {
  state = {
    width: window.innerWidth,
  };

  componentWillMount() {
    window.addEventListener('resize', this.handleWindowSizeChange);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.handleWindowSizeChange);
  }

  handleWindowSizeChange = () => {
    this.setState({ width: window.innerWidth });
  };

  render() {
    const { width } = this.state;
    const window_width = (width < 500) ? 300 : 520;
    const window_height = (window_width == 500) ? 240*2 : 240*4;
    const inner_style = {
      margin: "0 Auto",
      height: window_height,
      width: window_width,
      position:'relative',
    }
    return (
      <div style={styles.outer}>
        <Header loggedIn={true} />
        <div style={inner_style}>
          <Link to={"/articles"}>
            <Card style={styles.card}>
              <CardMedia
                image="/static/frontend/newspapers_xl_28444782.jpg"
                style={styles.media}
                title="Contemplative Reptile"
              />
              <CardContent>
                <Typography varient="headline" style={styles.title}>
                  Articles
                </Typography>
              </CardContent>
            </Card>
          </Link>
          <Link to={"/player_articles"}>
            <Card style={styles.card}>
              <CardMedia
                image="/static/frontend/football.jpg"
                style={styles.media}
                title="Contemplative Reptile"
              />
              <CardContent>
                <Typography varient="headline" style={styles.title}>
                  Classification
                </Typography>
              </CardContent>
            </Card>
          </Link>
          <Link to={"/player_info"}>
            <Card style={styles.card}>
              <CardMedia
                image="/static/frontend/classify.jpeg"
                style={styles.media}
                title="Contemplative Reptile"
              />
              <CardContent>
                <Typography varient="headline" style={styles.title}>
                  Fetch articles with keyword
                </Typography>
              </CardContent>
            </Card>
          </Link>
          <Link to={"/fast_classify"}>
            <Card style={styles.card}>
              <CardMedia
                image="/static/frontend/classify.jpeg"
                style={styles.media}
                title="Contemplative Reptile"
              />
              <CardContent>
                <Typography varient="headline" style={styles.title}>
                  Fetch paragraphs with keyword
                </Typography>
              </CardContent>
            </Card>
          </Link>
        </div>
      </div>
    );
  }
}

export default Home;
