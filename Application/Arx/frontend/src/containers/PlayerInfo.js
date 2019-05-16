import React, { Component } from "react";
import PropTypes from "prop-types";
import TextField from '@material-ui/core/TextField';

import ArticleTagList from "./../components/ArticleTagList";
import Header from "./../components/Header";
import LoginPage from "./../components/LoginPage"


const styles = {
  textField: {
    bottomMargin: 10,
  },
  content: {
    position:'absolute',
    width: "100%",
  },
};

class PlayerInfo extends Component {
  state = {
    name: '',
    data: [],
    loaded: false,
    loggedIn: true,
    placeholder: "Search for player or club",
  };


  handleChange = event => {
    this.setState({
      name: event.target.value,
    });
  };

  catchReturn = (ev) => {
    if (ev.key !== "Enter"){
      return;
    }
    const searchTerm = this.state.name.replace(" ", "_");
    const path = "api/player_articles/" + searchTerm;
    const csrftoken = document.cookie.split("=")[1];

    fetch(path, {
      method: "GET",
      headers: {
        "X-CSRFToken": csrftoken,
      },
      credentials: "include",
    })
      .then(response => {
        if (response.status === 200) {
          this.setState({ loggedIn: true });
          return response.json();
        }
        else if (response.status === 403) {
          this.setState({ loggedIn: false });
          return 1;
        }
        this.setState({ placeholder: "Something went wrong" });
        return 1;
      })
      .then(data => {
        if (data === 1){
          return;
        }
        this.setState({ data: data, loaded: true })
      });
  };

  render() {
    const { data, loaded, loggedIn, placeholder } = this.state;
    if (!loggedIn){
      return (
        <div>
          <Header loggedIn={loggedIn} />
          <LoginPage />
        </div>
      );
    }

    return loaded ?
      <div>
        <Header loggedIn={loggedIn} />
        <TextField
          id="standard-name"
          label="Search"
          value={this.state.name}
          style={styles.textField}
          fullWidth={true}
          margin="normal"
          onChange={this.handleChange}
          onKeyPress={this.catchReturn}
        />
        <ArticleTagList data={data} />
      </div>
    :
      <div>
        <Header loggedIn={loggedIn} />
        <TextField
          id="standard-name"
          label="Search"
          value={this.state.name}
          style={styles.textField}
          fullWidth={true}
          margin="normal"
          onChange={this.handleChange}
          onKeyPress={this.catchReturn}
        />
        <p>{this.state.placeholder}</p>
      </div>
  }
}

export default PlayerInfo;
