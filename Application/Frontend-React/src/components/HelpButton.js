import React, { Component } from "react";
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';

class HelpButton extends Component {
  state = {
    open: false,
  };

  handleClickOpen = () => {
    this.setState({ open: true });
  };

  handleClose = () => {
    this.setState({ open: false });
  };
  render(){
    return (
      <div>
        <Button onClick={this.handleClickOpen}>Help</Button>
        <Dialog
          open={this.state.open}
          onClose={this.handleClose}
          aria-labelledby="alert-dialog-title"
          aria-describedby="alert-dialog-description"
        >
          <DialogTitle id="alert-dialog-title">{"Help"}</DialogTitle>
          <DialogContent>
              <p>
                This page is used to make data sample for my master thesis. Each
                paragraph is one data sample. Beside each paragraph, there is an
                icon that will display a list of possible labels when pressed.
                Choose the label that fits best with the paragraph. If you are
                uncertain, skip it.
              </p>
              <br/>
              <p>
                Here is a description of the different classes
              </p>
              <strong> Ignore </strong>
              <p>
                experts opinions, short irrelevant paragraph, reporter questions.
              </p>
              <strong> Transfer </strong>
              <p>
                Any transfer rumours regarding a player. Speculations about a
                players future.
              </p>
              <strong> Goal/Assist </strong>
              <p>
                Goals or assist done by a player, or goal/assist descriptions.
              </p>
              <strong> Club drama </strong>
              <p>
                Any conflict a player may have with the club, manager, or other
                players in the club.
              </p>
              <strong> Personal drama </strong>
              <p>
                Any conflict that revolves around a players life, for example,
                relationship, family, players from other clubs, DUI etc.
              </p>
              <strong> Injuries </strong>
              <p>
                Any injuries related to a player.
              </p>
              <br/>
              <p>
                If you choose a wrong label by accident, you can press the option
                "Pressed wrong button", this will reset the label for the paragraph.
                However, if you have reloaded the page, the change is permanent.
              </p>
              <br/>
              <p>
                When all the paragraph, or at least all the relaven paragraphes,
                are classified, press the classified button at the bottom of the
                page.
              </p>
              <p>
                Any questions or problems send mail to aanundn@gmail.com
              </p>
          </DialogContent>
          <DialogActions>
            <Button onClick={this.handleClose} color="primary">
              OK
            </Button>
          </DialogActions>
        </Dialog>
      </div>
    );
  }
}

export default HelpButton;
