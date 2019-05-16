import React, { Component } from "react";
import PropTypes from "prop-types";
import Button from '@material-ui/core/Button';

class LogoutButton extends Component {
  handleClick = () => {
    fetch("accounts/logout/")
      .then(response => {
        if (response.status === 200){
          return response.json();
        }
        else{
          return 1;
        }
      })
      .then(data => {
        return data
      });
  }
  render(){
    const { loggedIn } = this.props;
    if (loggedIn){
      return (
        <Button style={{float:"right"}} onClick={this.handleClick}>
          Log out
        </Button>
      );
    }

    return null;
  }
}

LogoutButton.propTypes = {
  loggedIn: PropTypes.bool.isRequired,
};

export default LogoutButton;
