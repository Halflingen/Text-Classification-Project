import React, { Component } from "react";
import PropTypes from "prop-types";
import { Link } from "react-router-dom"
import Button from '@material-ui/core/Button';

import ArticleContent from "./ArticleContent"

const style = {
  float: "right",
  backgroundColor: "rgb(63, 81, 181)",
}

class PageDisplay extends Component {

  handleClick = (id_) => {
    const classification = {classified: true};
    const path = "api/article_info/" + id_;
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
    .then(response => response.json())
    .then(data => data);
  };

  render(){
    const { data } = this.props;
    const id = data[0].article_id;
    return (
      <div style={{position:'absolute'}}>
        <ArticleContent data={data} />
        <Button style={style} onClick={() => this.handleClick(id)}>Classified</Button>
      </div>
    );
  }
}

PageDisplay.propTypes = {
  data: PropTypes.array.isRequired,
};

export default PageDisplay;
