import React, { Component } from "react";
import PropTypes from "prop-types";
import Typography from '@material-ui/core/Typography';

import ClassifyIcon from "./ClassifyIcon";

const style = {
  headline: {
    fontSize: "200%",
    //border: "1px solid black",
  },
  description: {
    fontSize: 16, //same as 100%
    //border: "1px solid black",
  },
  paragraph: {
    fontSize: 16, //same as 100%
    //border: "1px solid black",
  },
  container: {
    width: "100%",
    //border: "1px solid black",
  },
}

class ClassifyParagraph extends Component {
  render(){
    const { content, id, content_order, html_type, class_field } = this.props;
    var body = null;
    if (content_order == 0){
      body = (
        <Typography key={id} paragraph={true} variant="title" style={style.headline}>
          {content}
        </Typography>
      )
    }
    else if (content_order == 1){
      body = (
        <Typography key={id} paragraph={true} style={style.description}>
          <strong>{content}</strong>
        </Typography>
      )
    }
    else {
      body = (
        <Typography key={id} paragraph={true} style={style.paragraph}>
          {content}
        </Typography>
      )
    }
    const classButton = class_field ? null :
          <ClassifyIcon id={id} content_order={content_order} />;

    return (
      <div style={style.container}>
        {classButton}
        {body}
      </div>
    );
  }
}


ClassifyParagraph.propTypes = {
  content: PropTypes.string.isRequired,
  id: PropTypes.string.isRequired,
  content_order: PropTypes.number.isRequired,
  html_type: PropTypes.string.isRequired,
  class_field: PropTypes.string,
};

export default ClassifyParagraph;
