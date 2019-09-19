import React, { Component } from "react";
import PropTypes from "prop-types";
import { Link } from 'react-router-dom';

import Typography from '@material-ui/core/Typography';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';

class ArticleTag extends Component {

  render(){
    const { published, article_id, headline, source } = this.props;
    const path = "/article/" + article_id;
    const date = published.split("T")[0];
    return(
      <Link to={path}>
        <Card style={{width:"100%", marginBottom:3}}>
          <CardContent>
            <Typography  variant="title" component="h2">
              {headline}
            </Typography>
            <Typography  style={{float:"left"}} variant="caption">
              {source}
            </Typography>
            <Typography style={{float:"right"}}  variant="caption">
              {date}
            </Typography>
          </CardContent>
        </Card>
      </Link>
    );
  }
}


ArticleTag.propTypes = {
  headline: PropTypes.string.isRequired,
  article_id: PropTypes.string.isRequired,
  source: PropTypes.string.isRequired,
  published: PropTypes.string.isRequired,
};

export default ArticleTag;
