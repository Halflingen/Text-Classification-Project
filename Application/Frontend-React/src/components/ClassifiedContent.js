import React, { Component } from "react";
import PropTypes from "prop-types";
import Typography from '@material-ui/core/Typography';

class ClassifiedContent extends Component {
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

    /*
    const body = data.map(obj => (
        <Typography
          style={{fontSize:16}}
          paragraph={true}
        >
            {obj.content}
        </Typography>
    ));
    */
    const body = data.map(obj => (
        obj.label === obj.class?
        <Typography
          style={{fontSize:16, color:'green'}}
          paragraph={true}
        >
            {obj.content}
        </Typography>
      :
        <Typography
          style={{fontSize:16, color:'black'}}
          paragraph={true}
        >
            {obj.content}
        </Typography>
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

ClassifiedContent.propTypes = {
  data: PropTypes.array.isRequired,
};

export default ClassifiedContent;
