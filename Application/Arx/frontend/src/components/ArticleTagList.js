import React, { Component } from "react";
import PropTypes from "prop-types";

import ArticleTag from "./ArticleTag";

class ArticleTagList extends Component {

  render(){
    const { data } = this.props;
    return (
      <div style={{position:'absolute', width: "100%"}}>
        {data.map(obj => (
          <ArticleTag
            key={obj.article_id}
            headline={obj.headline}
            article_id={obj.article_id}
            source={obj.source}
            published={obj.published}
          />
        ))}
      </div>
    );
  }
}


ArticleTagList.propTypes = {
  data: PropTypes.array.isRequired,
};

export default ArticleTagList;
