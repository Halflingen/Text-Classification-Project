import React, { Component } from "react";
import PropTypes from "prop-types";

import classnames from 'classnames';
import Card from '@material-ui/core/Card';
import CardHeader from '@material-ui/core/CardHeader';
import CardMedia from '@material-ui/core/CardMedia';
import CardContent from '@material-ui/core/CardContent';
import CardActions from '@material-ui/core/CardActions';
import Collapse from '@material-ui/core/Collapse';
import Avatar from '@material-ui/core/Avatar';
import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';
import red from '@material-ui/core/colors/red';
import FavoriteIcon from '@material-ui/icons/Favorite';
import ShareIcon from '@material-ui/icons/Share';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import MoreVertIcon from '@material-ui/icons/MoreVert';

import CardActionArea from '@material-ui/core/CardActionArea';

import ClassifiedContent from './ClassifiedContent'

class PlayerBody extends Component {
  state = { expanded: false };

  handleExpandClick = () => {
    this.setState(state => ({ expanded: !state.expanded }));
  };

  render(){
    const { data, name } = this.props;
    const number_of_lable = data.length
    return (
      <div style={{width: "100%", margin:4}}>
      <Card style={{width: "100%"}}>
        <CardActionArea onClick={this.handleExpandClick} style={{width: "100%", height:80}}>
          <Typography variant="title" style={{float:'left'}}>
            {name}
          </Typography>
          <Typography variant="caption" style={{margin:4, float:'left'}}>
            ({number_of_lable})
          </Typography>
        </CardActionArea>
        <Collapse in={this.state.expanded} timeout="auto" unmountOnExit>
          <CardContent>
            <ClassifiedContent data={data} />
          </CardContent>
        </Collapse>
      </Card>
      </div>
    );
  }
}


//ArticleTagList.propTypes = {
  //data: PropTypes.array.isRequired,
//};

export default PlayerBody;
