import React, { Component } from "react";
import PropTypes from "prop-types";

const style = {
  display: "table-cell",
  textAlign:"center",
  verticalAlign: "middle",
};

class LoginPage extends Component {
  state = {
    username: "",
    password: "",
  };

  handleUserChange = (evt) => {
    this.setState({
      username: evt.target.value,
    });
  };

  handlePassChange = (evt) => {
    this.setState({
      password: evt.target.value,
    });
  }

  render(){
    const csrf_token = document.cookie.split("=")[1];
    const height = window.innerHeight;
    return (
      <div style={{width:100+"%", display: "table", height:height}}>
        <form action="accounts/login/" method="post" style={style}>
          <input type="hidden" name="csrfmiddlewaretoken" value={csrf_token} />
          <input
            id="id_username"
            name="username"
            placeholder="username"
            value={this.state.username}
            onChange={this.handleUserChange}
          />

          <input
            id="id_password"
            name="password"
            placeholder="password"
            type="password"
            value={this.state.password}
            onChange={this.handlePassChange}
          />

          <input type="submit" value="Log In" data-test="submit" />
        </form>
      </div>
    );
  }
}

export default LoginPage;
