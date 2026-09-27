# Upload to GitHub

## Using GitHub in a browser

1. Create a new empty repository on GitHub. Do not add a README or license there because this package already contains both.
2. Extract `HCE_GitHub_Reproducibility_v1.0.0.zip`.
3. Upload the contents of the extracted folder, not the ZIP as a single repository file.
4. After the files are visible, create the release tag `v1.0.0` and attach the ZIP as a release asset.

## Using Git

Run the following commands inside the extracted folder, replacing the placeholder URL with the empty repository URL created on GitHub:

```bash
git init
git add .
git commit -m "Release reproducibility package v1.0.0"
git branch -M main
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
git tag -a v1.0.0 -m "HCE/EAGB reproducibility release v1.0.0"
git push origin v1.0.0
```

Before making the repository public, check `RELEASE_CHECKLIST.md`. Do not upload the excluded `data/raw/` directory or policy-/claim-level files.
