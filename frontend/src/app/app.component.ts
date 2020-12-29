import { Component } from '@angular/core';
import { FormGroup, FormBuilder, Validators, FormControl } from '@angular/forms';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'News Bites';
  inputForm: FormGroup;
  submitted : boolean;

  constructor(private fb: FormBuilder) {
    this.createForm();
  }

  createForm() {
    this.inputForm = this.fb.group({
      url:['', Validators.required]
    });
  }

  onSubmit() {
    this.submitted = true;
    if(this.inputForm.valid) {
      //submit
    }
  }
}
