import React, { Component } from "react";
import PropTypes from "prop-types";

import IconButton from "@material-ui/core/IconButton";
import Menu from "@material-ui/core/Menu";
import MenuItem from "@material-ui/core/MenuItem";
import MoreVertIcon from "@material-ui/icons/MoreVert";

const options = [
  'Quote',
  'Irrelevant',
  'Ignore',
  'Transfer',
  'Goal/Assist',
  'Club details',
  'Player details',
  'Chance',
];


const style = {
  IconContainer: {
    //display: "inline-block",
    //border: "1px solid black",
    float: "right",
    margin: -10,
  },
}

const ITEM_HEIGHT = 48;

class ClassifyIcon extends Component {
  state = {
    anchorEl: null,
    showIcon: true,
  };

  handleClick = event => {
    this.setState({ anchorEl: event.currentTarget });
  };

  handleClose = (option, id_, content_order) => {
    this.setState({ anchorEl: null });
  };

  handleMenuChoise = (option, id_, content_order) => {
    if (option == "Pressed wrong button"){
      option = null;
    }
    const classification = {class_field: option};
    const path = "api/article_content/" + id_ + "/" + content_order;
    const csrftoken = document.cookie.split("=")[1];

    fetch(path, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "X-CSRFToken": csrftoken,
      },
      credentials: "include",
      body: JSON.stringify(classification),
    })
      .then(response => {
        if (response.status === 200){
          return response.json();
        }
        else if (response.status === 403){
          alert("You must be logged in. Refresh the page");
          return 1;
        }
        alert("Something went wrong. The request did not go through");
        return 1;
      })
      .then(data => data);

    if (option == "Pressed wrong button"){
      this.setState({ anchorEl: null, showIcon: true });
      return;
    }
    this.setState({ anchorEl: null, showIcon: false });
  }

  render(){
    const { id, content_order } = this.props;

    const { anchorEl, showIcon } = this.state;
    const open = Boolean(anchorEl);
    return (
      <div style={style.IconContainer}>
        <IconButton
          aria-label="More"
          aria-owns={open ? 'long-menu' : null}
          aria-haspopup="true"
          onClick={this.handleClick}
        >
          {showIcon ? <MoreVertIcon /> : null}
        </IconButton>
        <Menu
          id="long-menu"
          anchorEl={anchorEl}
          open={open}
          onClose={this.handleClose}
        >
          {options.map(option => (
            <MenuItem key={option} onClick={() => this.handleMenuChoise(option, id, content_order)}>
              {option}
            </MenuItem>
          ))}
        </Menu>
      </div>
    );
  }
}


ClassifyIcon.propTypes = {
  id: PropTypes.string.isRequired,
  content_order: PropTypes.number.isRequired,
};

export default ClassifyIcon;
