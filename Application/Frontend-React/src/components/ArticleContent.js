import React, { Component } from "react";
import PropTypes from "prop-types";
import Typography from '@material-ui/core/Typography';

import ClassifyParagraph from "./ClassifyParagraph"

class ArticleContent extends Component {
  state = {
    width: window.innerWidth,
  };

  componentWillMount() {
    window.addEventListener('resize', this.handleWindowSizeChange);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.handleWindowSizeChange);
  }

  handleWindowSizeChange = () => {
    this.setState({ width: window.innerWidth });
  };

  render() {
    const { data } = this.props;

    const body = data.map(obj => (
      (obj.html_type == "h2" && obj.content_order != 1) ?
        <Typography
          key={obj.id}
          style={{fontSize:"150%"}}
        >
            <strong>{obj.content}</strong>
        </Typography>
        :
        <ClassifyParagraph
          key={obj.id}
          content={obj.content}
          id={obj.article_id}
          content_order={obj.content_order}
          html_type={obj.html_type}
          class_field={obj.class_field}
        />
    ));

    const { width } = this.state;
    const widthPercent = (width < 500) ? "100%" : ((500/width)*100)+"%";

    return (
      <div style={{margin:"0 auto", width: widthPercent}}>
        {body}
      </div>
    );
  }
}

ArticleContent.propTypes = {
  data: PropTypes.array.isRequired,
};

export default ArticleContent;
