import React, { Component } from "react";
import PropTypes from "prop-types";

import Header from "./../components/Header";
import PageDisplay from "./../components/PageDisplay"
import LoginPage from "./../components/LoginPage"

class ContentPage extends Component {
  state = {
    data: [],
    loaded: false,
    loggedIn: true,
    placeholder: "Loading...",
  };

  componentDidMount() {
    const parameter = this.props.match.params;
    const path = "api/article_content/" + parameter.article_id
    fetch(path, {credentials: "include"})
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
  }

  render(){
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
        <PageDisplay data={data} />
      </div>
    :
      <div>
        <Header loggedIn={loggedIn} />
        <p>{this.state.placeholder}</p>
      </div>
  }
}

export default ContentPage;
