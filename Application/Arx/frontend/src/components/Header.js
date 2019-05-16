import React, { Component } from "react";
import PropTypes from "prop-types";
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import IconButton from "@material-ui/core/IconButton";
import MenuIcon from '@material-ui/icons/Menu';
import Menu from "@material-ui/core/Menu";
import MenuItem from "@material-ui/core/MenuItem";
import { Link } from "react-router-dom";

import HelpButton from "./HelpButton";
import LogoutButton from "./LogoutButton";

const styles = {
  textField: {
    margin: "0 auto",
  },
  content: {
    position:'absolute',
    width: "100%",
  },
  Button: {
    backgroundColor: "rgb(63, 81, 181)",
  }
};

const options = [
  'Help',
  'Log Out',
  'label count',
];

class Header extends Component {
  state = {
    anchorEl: null,
  };

  handleClick = event => {
    this.setState({ anchorEl: event.currentTarget });
  };

  handleClose = (option, id_, content_order) => {
    this.setState({ anchorEl: null });
  };

  handleMenuChoise = (option) => {
     if (option == 'label count'){

     }
     this.setState({ anchorEl: null });
  }

  render(){
    const { loggedIn } = this.props;
    const { anchorEl } = this.state;
    const open = Boolean(anchorEl);
    return (
      <AppBar position="static" style={{marginBottom:20}}>
        <Toolbar>
          <Link to="/" style={{margin:"0 auto"}}>
            <Typography variant="title">
              ARX
            </Typography>
          </Link>
        <IconButton
          aria-label="More"
          aria-owns={open ? 'long-menu' : null}
          aria-haspopup="true"
          onClick={this.handleClick}
        >
          <MenuIcon />
        </IconButton>
        <Menu
          id="long-menu"
          anchorEl={anchorEl}
          open={open}
          onClose={this.handleClose}
        >
          {options.map(option => (
            <MenuItem key={option} onClick={() => this.handleMenuChoise(option)}>
              {option}
            </MenuItem>
          ))}
        </Menu>
        </Toolbar>
      </AppBar>
    );
  }
}

Header.propTypes = {
  loggedIn: PropTypes.bool.isRequired,
};

export default Header;
