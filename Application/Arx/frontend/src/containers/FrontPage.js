import React, { Component } from "react";
import PropTypes from "prop-types";

import ArticleTagList from "./../components/ArticleTagList";
import Header from "./../components/Header";
import LoginPage from "./../components/LoginPage"

class FrontPage extends Component {
  state = {
    data: [],
    loaded: false,
    loggedIn: true,
    placeholder: "Loading...",
  };

  componentDidMount() {
    fetch("api/article_info/", {credentials: "include"})
      .then(response => {
        if (response.status === 200) {
          this.setState({ loggedIn: true });
          return response.json();
        }
        else if (response.status === 403){
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
        this.setState({ data: data, loaded: true });
      });
  }

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
        <ArticleTagList data={data} />
      </div>
    :
      <div>
        <Header loggedIn={loggedIn} />
        <p>{this.state.placeholder}</p>
      </div>
  }
}

export default FrontPage;
