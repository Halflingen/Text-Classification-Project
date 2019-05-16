import React, { Component } from "react";
import PropTypes from "prop-types";
import TextField from '@material-ui/core/TextField';

import Header from "./../components/Header";
import ArticleContent from "./../components/ArticleContent"

const styles = {
  textField: {
    bottomMargin: 10,
  },
  content: {
    position:'absolute',
    width: "100%",
  },
};

class FastClassify extends Component {
   state = {
    name: '',
    data: [],
    loaded: false,
    loggedIn: true,
    placeholder: "Loading...",
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
    const path = "api/fast_classify/" + searchTerm;
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
    const { data } = this.state;
    return (
      <div style={styles.content}>
        <Header loggedIn={true} />
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
        <ArticleContent data={data} />
      </div>
    );
  }
}

export default FastClassify;
