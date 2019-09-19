import React, { Component } from "react";
import PropTypes from "prop-types";
import TextField from '@material-ui/core/TextField';

import ArticleTagList from "./../components/ArticleTagList";
import Header from "./../components/Header";
import LoginPage from "./../components/LoginPage"
import PlayerBody from "./../components/PlayerBody"
import Radio from '@material-ui/core/Radio';
import RadioButtonUncheckedIcon from '@material-ui/icons/RadioButtonUnchecked';
import RadioButtonCheckedIcon from '@material-ui/icons/RadioButtonChecked';


const styles = {
  textField: {
    bottomMargin: 10,
  },
  content: {
    position:'absolute',
    width: "100%",
  },
};

class PlayerArticles extends Component {
  state = {
    name: '',
    data: [],
    loaded: false,
    loggedIn: true,
    placeholder: "Search for player or club",
    selectedValue: "cnn",
  };


  handleChangeSearch = event => {
    this.setState({
      name: event.target.value,
    });
  };

  handleChangeModel = event => {
    this.setState({
      selectedValue: event.target.value,
    });
    if (this.state.name == ''){
        return
    }
    this.fetch_paragraphs()
  };

  catchReturn = (ev) => {
    if (ev.key !== "Enter"){
      return;
    }
    this.fetch_paragraphs()
  };

  fetch_paragraphs(){
    const searchTerm = this.state.name.replace(" ", "_");
    const path = "api/player_classifier/" + this.state.selectedValue + '/' + searchTerm;
    const csrftoken = document.cookie.split("=")[1];

    fetch(path, {
      method: "GET",
      headers: {
        "X-CSRFToken": csrftoken,
      },
      credentials: "include",
    })
      .then(response => {
        if (response.status === 200) {
          this.setState({ loggedIn: true });
          return response.json();
        }
        else if (response.status === 403) {
          this.setState({ loggedIn: false });
          return 1;
        }
        this.setState({ placeholder: "Something went wrong" });
        return 1;
      })
      .then(data => {
        if (data === 1){
          return;
        }
        this.setState({ data: data, loaded: true })
      });
  }

  render() {
    const { data, loaded, loggedIn, placeholder } = this.state;
    if (!loggedIn){
      return (
        <div>
          <Header loggedIn={loggedIn} />
          <LoginPage />
        </div>
      );
    }
    if (loaded){
        const data_1 = data['data'].filter(obj => obj.label=='1')
        const data_2 = data['data'].filter(obj => obj.label=='2')
        const data_3 = data['data'].filter(obj => obj.label=='3')
        const data_4 = data['data'].filter(obj => obj.label=='4')
        const data_5 = data['data'].filter(obj => obj.label=='0')
        return (
          <div>
            <Header loggedIn={loggedIn} />
            <TextField
              id="standard-name"
              label="Search"
              value={this.state.name}
              style={styles.textField}
              fullWidth={true}
              margin="normal"
              onChange={this.handleChangeSearch}
              onKeyPress={this.catchReturn}
            />
          <div>
            <Radio
              checked={this.state.selectedValue === 'cnn'}
              onChange={this.handleChangeModel}
              value="cnn"
              color="default"
              name="radio-button-demo"
              aria-label="A"
            />
            CNN
            <Radio
              checked={this.state.selectedValue === 'rnn'}
              onChange={this.handleChangeModel}
              value="rnn"
              color="default"
              name="radio-button-demo"
              aria-label="B"
            />
            RNN
            <Radio
              checked={this.state.selectedValue === 'linearSVC'}
              onChange={this.handleChangeModel}
              value="linearSVC"
              color="default"
              name="radio-button-demo"
              aria-label="C"
            />
            Linear SVC
            <Radio
              checked={this.state.selectedValue === 'svc_poly'}
              onChange={this.handleChangeModel}
              value="svc_poly"
              color="default"
              name="radio-button-demo"
              aria-label="D"
            />
            SVC Poly
            <Radio
              checked={this.state.selectedValue === 'svc_rbf'}
              onChange={this.handleChangeModel}
              value="svc_rbf"
              color="default"
              name="radio-button-demo"
              aria-label="E"
            />
            SVC RBF
            <Radio
              checked={this.state.selectedValue === 'nb'}
              onChange={this.handleChangeModel}
              value="nb"
              color="default"
              name="radio-button-demo"
              aria-label="F"
            />
            Naive Bayes
          </div>
            <PlayerBody data={data_1} name="Goal/Assist"/>
            <PlayerBody data={data_2} name="Transfer"/>
            <PlayerBody data={data_3} name="Quotes"/>
            <PlayerBody data={data_4} name="Irrelevant"/>
            <PlayerBody data={data_5} name="Ignore"/>
          </div>
        )
    }

    return (
      <div>
        <Header loggedIn={loggedIn} />
        <TextField
          id="standard-name"
          label="Search"
          value={this.state.name}
          style={styles.textField}
          fullWidth={true}
          margin="normal"
          onChange={this.handleChangeSearch}
          onKeyPress={this.catchReturn}
        />
      <div>
        <Radio
          checked={this.state.selectedValue === 'cnn'}
          onChange={this.handleChangeModel}
          value="cnn"
          color="default"
          name="radio-button-demo"
          aria-label="A"
        />
        CNN
        <Radio
          checked={this.state.selectedValue === 'rnn'}
          onChange={this.handleChangeModel}
          value="rnn"
          color="default"
          name="radio-button-demo"
          aria-label="B"
        />
        RNN
        <Radio
          checked={this.state.selectedValue === 'linearSVC'}
          onChange={this.handleChangeModel}
          value="linearSVC"
          color="default"
          name="radio-button-demo"
          aria-label="C"
        />
        Linear SVC
        <Radio
          checked={this.state.selectedValue === 'svc_poly'}
          onChange={this.handleChangeModel}
          value="svc_poly"
          color="default"
          name="radio-button-demo"
          aria-label="D"
        />
        SVC Poly
        <Radio
          checked={this.state.selectedValue === 'svc_rbf'}
          onChange={this.handleChangeModel}
          value="svc_rbf"
          color="default"
          name="radio-button-demo"
          aria-label="E"
        />
        SVC RBF
        <Radio
          checked={this.state.selectedValue === 'nb'}
          onChange={this.handleChangeModel}
          value="nb"
          color="default"
          name="radio-button-demo"
          aria-label="F"
        />
        Naive Bayes
      </div>
        <p>{this.state.placeholder}</p>
      </div>
    )
  }
}

export default PlayerArticles;
